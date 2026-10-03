// sdbench: Mac head-only microbench for split decode. Loads the model exactly as serve.sh SPLIT_DECODE_L does (llama-server's
// CLI args: -ot layers >= L and output to CPU, --no-repack, mmap; LLAMA_SPLIT_DECODE=1 + LLAMA_SPLIT_L so P9 memory and the head
// graph [0, L)), but the "phone" is an in-process stub tail on loopback that acks every chunk at once with zero logits.
// Then times N single-token decodes. Env:
//   SDB_PROMPT  text file (special tokens parsed)    SDB_N  decodes (200)    SDB_WARM  untimed decodes first (5)
//   SDB_OUT     per-token CSV (optional)             SDB_DEPTH  pad the prompt to this many tokens (repeats it)
// Per decode: wall, and from the split verbose line "Mac X ms, wait Y ms". With GGML_METAL_TIMELINE=<file> the timeline is
// parsed afterwards by sdbench.py (the bench writes its own decode start/end marks into SDB_MARKS, same mach clock).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "tail-server.h"

#include <mach/mach_time.h>
#include <sys/time.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static double now_s() {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double) mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

// ---- stub tail ------------------------------------------------------------------------------------------------------------
static int g_listen = -1;
static int g_port = 0;
static std::string g_desc;   // set before the model loads? no: filled from the HELLO's n_vocab etc.; desc from the model below
static std::atomic<bool> g_stop{false};

static void stub_serve() {
    using namespace spt;
    while (!g_stop) {
        int fd = accept(g_listen, nullptr, nullptr);
        if (fd < 0) continue;
        tune_socket(fd);
        uint32_t n_valid = 0, n_vocab = 0, n_embd = 0;
        std::vector<float> zeros;
        std::vector<char> buf;
        try {
            for (;;) {
                msg_hdr h = recv_hdr(fd);
                buf.resize(h.len);
                if (h.len) recv_all(fd, buf.data(), h.len);
                if (h.type == MSG_HELLO) {
                    hello_req2 q; memcpy(&q, buf.data(), std::min(sizeof(q), buf.size()));
                    hello_rep2 r = {};
                    r.base.proto = PROTO_VERSION; r.base.state_format = STATE_FORMAT;
                    r.base.layer_start = q.base.L; r.base.n_layer_full = q.base.n_layer_full; r.base.n_layer = q.base.n_layer_full - q.base.L;
                    r.base.n_embd = n_embd = q.base.n_embd; r.base.n_vocab = n_vocab = q.base.n_vocab; r.base.n_ctx = q.base.n_ctx;
                    snprintf(r.base.desc, sizeof(r.base.desc), "stub %s", g_desc.c_str());
                    r.n_valid = n_valid = 0; r.session = q.session;
                    zeros.assign(n_vocab, 0.0f);
                    send_msg(fd, MSG_HELLO_OK, &r, sizeof(r));
                } else if (h.type == MSG_TRIM) {
                    uint32_t n; memcpy(&n, buf.data(), 4);
                    n_valid = n < n_valid ? 0 : n_valid;
                    send_msg(fd, MSG_OK, &n_valid, 4);
                } else if (h.type == MSG_CHUNK) {
                    chunk_req q; memcpy(&q, buf.data(), sizeof(q));
                    n_valid += q.n_tok;
                    static const int stub_ms = getenv("SDB_STUB_MS") ? atoi(getenv("SDB_STUB_MS")) : 0;   // the phone's compute + link
                    if (stub_ms > 0 && q.n_tok == 1) usleep(stub_ms * 1000);
                    chunk_rep r = { 0, q.n_tok, (q.want_logits & 1) ? n_vocab : 0, 0.0f };
                    send_msg(fd, MSG_CHUNK_ACK, &r, sizeof(r), zeros.data(), r.n_logits * sizeof(float));
                } else if (h.type == MSG_BYE) {
                    break;
                } else {
                    const char * e = "stub: unsupported";
                    send_msg(fd, MSG_ERR, e, strlen(e));
                }
            }
        } catch (const std::exception & e) {
            if (!g_stop) fprintf(stderr, "stub: %s\n", e.what());
        }
        close(fd);
    }
}

// ---- split verbose lines --------------------------------------------------------------------------------------------------
static std::mutex g_mu;
static std::vector<std::pair<double, double>> g_split;   // Mac ms, wait ms
static void log_cb(ggml_log_level level, const char * text, void * ud) {
    const char * p = strstr(text, "split decode 1 tokens at");
    if (p) {
        const char * m = strstr(p, "Mac ");
        const char * w = strstr(p, "wait ");
        if (m && w) {
            std::lock_guard<std::mutex> lk(g_mu);
            g_split.push_back({ atof(m + 4), atof(w + 5) });
        }
        return;
    }
    if (strstr(text, "METAL-PROF") || strstr(text, "METAL-BYTES") || level >= GGML_LOG_LEVEL_WARN || getenv("SDB_LOG")) {
        fputs(text, stderr);
    }
    (void) ud;
}

static double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5))];
}

int main(int argc, char ** argv) {
    // stub listener first: LLAMA_SPLIT_TAIL must point at it before the context is created
    g_listen = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(g_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
    if (bind(g_listen, (sockaddr *) &a, sizeof(a)) || listen(g_listen, 4)) { perror("stub bind"); return 1; }
    socklen_t al = sizeof(a); getsockname(g_listen, (sockaddr *) &a, &al); g_port = ntohs(a.sin_port);
    const std::string tail = "127.0.0.1:" + std::to_string(g_port);
    setenv("LLAMA_SPLIT_TAIL", tail.c_str(), 1);
    setenv("LLAMA_SPLIT_DECODE", "1", 1);
    if (!getenv("LLAMA_SPLIT_L")) setenv("LLAMA_SPLIT_L", "20", 1);
    setenv("LLAMA_SPLIT_VERBOSE", "1", 1);

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) return 1;
    common_init();
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    // the split client checks that the worker desc contains this model's ftype text: claim the two quants benched here
    g_desc = "IQ2_XS - 2.3125 bpw | IQ2_M - 2.7 bpw";
    std::thread stub(stub_serve);

    const double t_load0 = now_s();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "sdbench: init failed\n"); return 2; }
    fprintf(stderr, "sdbench: loaded in %.1f s, stub tail %s, desc '%s'\n", now_s() - t_load0, tail.c_str(), g_desc.c_str());

    std::string ptxt = "The history of the printing press began";
    if (const char * pf = getenv("SDB_PROMPT")) { std::ifstream f(pf); std::stringstream ss; ss << f.rdbuf(); ptxt = ss.str(); }
    auto prompt = common_tokenize(ctx, ptxt, true, true);
    if (const char * dp = getenv("SDB_DEPTH")) {
        const size_t want = atoi(dp), n0 = prompt.size();
        while (prompt.size() < want) prompt.push_back(prompt[1 + (prompt.size() % (n0 - 1))]);
        prompt.resize(want);
    }
    const int N = getenv("SDB_N") ? atoi(getenv("SDB_N")) : 200;
    const int WARM = getenv("SDB_WARM") ? atoi(getenv("SDB_WARM")) : 5;

    llama_memory_clear(llama_get_memory(ctx), true);
    const int n_batch = llama_n_batch(ctx);
    double t0 = now_s();
    for (size_t i = 0; i < prompt.size(); i += n_batch) {
        const int n = (int) std::min<size_t>(n_batch, prompt.size() - i);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int j = 0; j < n; j++) common_batch_add(b, prompt[i + j], (llama_pos) (i + j), { 0 }, i + j == prompt.size() - 1);
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc) { fprintf(stderr, "sdbench: prompt decode failed rc=%d\n", rc); return 3; }
    }
    fprintf(stderr, "sdbench: prompt %zu tokens in %.0f ms\n", prompt.size(), (now_s() - t0) * 1e3);

    llama_perf_context_reset(ctx);
    { std::lock_guard<std::mutex> lk(g_mu); g_split.clear(); }
    FILE * marks = getenv("SDB_MARKS") ? fopen(getenv("SDB_MARKS"), "w") : nullptr;
    std::vector<double> wall;
    llama_pos pos = (llama_pos) prompt.size();
    llama_batch b = llama_batch_init(1, 0, 1);
    double u0 = 0;
    for (int s = 0; s < WARM + N; s++) {
        if (s == WARM) { timeval tv; gettimeofday(&tv, nullptr); u0 = tv.tv_sec + tv.tv_usec * 1e-6; }
        common_batch_clear(b);
        common_batch_add(b, prompt[1 + (s % (prompt.size() - 1))], pos++, { 0 }, true);
        const double ta = now_s();
        const int rc = llama_decode(ctx, b);
        const float * lg = llama_get_logits_ith(ctx, -1);   // as the server does (sampling reads the row)
        const double tb = now_s();
        if (rc || !lg) { fprintf(stderr, "sdbench: decode failed at %d rc=%d\n", s, rc); return 4; }
        if (s >= WARM) {
            wall.push_back((tb - ta) * 1e3);
            if (marks) fprintf(marks, "D %.6f %.6f\n", ta, tb);
        }
        if (const char * gap = getenv("SDB_GAP_MS")) usleep(atoi(gap) * 1000);   // optional pause between tokens
    }
    llama_batch_free(b);
    if (marks) fclose(marks);
    { timeval tv; gettimeofday(&tv, nullptr); printf("WINDOW %.3f %.3f\n", u0, tv.tv_sec + tv.tv_usec * 1e-6); }
    const auto perf = llama_perf_context(ctx);

    std::vector<double> mac, wait;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (size_t i = g_split.size() - std::min(g_split.size(), (size_t) N); i < g_split.size(); i++) {
            mac.push_back(g_split[i].first); wait.push_back(g_split[i].second);
        }
    }
    auto mean = [](const std::vector<double> & v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size(); };
    printf("RESULT n=%d wall mean %.2f p50 %.2f p95 %.2f min %.2f | Mac mean %.2f p50 %.2f p95 %.2f | wait mean %.2f p50 %.2f | reused %d/%d evals\n",
           N, mean(wall), pct(wall, .5), pct(wall, .95), pct(wall, 0), mean(mac), mac.empty() ? 0 : pct(mac, .5), mac.empty() ? 0 : pct(mac, .95),
           mean(wait), wait.empty() ? 0 : pct(wait, .5), perf.n_reused, perf.n_eval);
    if (const char * o = getenv("SDB_OUT")) {
        FILE * f = fopen(o, "w");
        fprintf(f, "i,wall_ms,mac_ms,wait_ms\n");
        for (size_t i = 0; i < wall.size(); i++) fprintf(f, "%zu,%.3f,%.3f,%.3f\n", i, wall[i], i < mac.size() ? mac[i] : 0, i < wait.size() ? wait[i] : 0);
        fclose(f);
    }
    fflush(stdout);
    g_stop = true;
    shutdown(g_listen, SHUT_RDWR); close(g_listen);
    _exit(0);   // skip teardown (the stub thread may sit in accept)
}
