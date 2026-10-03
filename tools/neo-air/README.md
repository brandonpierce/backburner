# neo-air tools

Bench and quality tools for split decode on an 8 GB Mac (MacBook Neo) + a phone tail (iPhone Air), as used for the
Phase B/C numbers. They link against an existing llama.cpp build and take llama-server's CLI arguments.

```bash
tools/neo-air/build.sh                       # -> tools/neo-air/bin/{sdref,sdbench,gpuwarm,therm}; BUILD=<dir> for another build
```

Output goes to `tools/neo-air/out/` (or `OUT=`). `bin/` and `out/` are git-ignored. `MODEL` defaults to
`~/Models/Qwen3.8-27B-IQ2_XS.gguf`.

## Same answers: `sdref` + `cmp.py`

`sdref` runs a prompt through the normal `llama_decode` path and dumps the full logits row at each of N steps (greedy, or
teacher-forced on a token file). `cmp.py` compares two dumps position by position: top-1 agreement and KL(ref ‖ test) over
the whole vocabulary.

```bash
# 1. Mac-only reference (no split engine). On 8 GB the CPU computes layers >= 20 from the mapped file: ~20 min per prompt.
OT=(-ot '^blk\.(20|[1-9][0-9][0-9]+|[3-9][0-9]|2[1-9])\.=CPU' -ot '^output=CPU' --no-repack)
for p in short code text; do tools/neo-air/run-ref.sh $p "${OT[@]}" -t 6 -tb 6; done

# 2. Split decode, teacher-forced on the reference's tokens (phone up, tail with the head loaded)
for p in short code text; do TAIL=169.254.x.y:50060 tools/neo-air/run-split.sh $p force; done

# 3. Compare (needs numpy)
for p in short code text; do uv run --with numpy tools/neo-air/cmp.py tools/neo-air/out/ref_$p tools/neo-air/out/sd_force_$p; done
```

`run-split.sh NAME greedy` is the free-running greedy output, for token-identity checks between builds
(`cmp -s out/a_greedy_short.tok out/b_greedy_short.tok`). Prompts: `prompts/p_short.txt` (19-token chat),
`p_code.txt` (1,059 tokens), `p_text.txt` (5,275 tokens).

## Mac head-only bench: `sdbench`

Loads the model as serve.sh `SPLIT_DECODE_L` does, with an in-process stub tail on loopback (zero logits), and times N
single-token decodes through the real split-decode path. No phone needed.

```bash
SDB_STUB_MS=160 LLAMA_SPLIT_GPU_WARM_US=1000 tools/neo-air/run-sdb.sh warm1ms     # stub holds each ack 160 ms (phone + link)
SDB_GAP_MS=160 tools/neo-air/run-sdb.sh gap                                       # or sleep between decodes instead
```

- Env: `SDB_N` (200), `SDB_WARM` (5), `SDB_PROMPT`, `SDB_DEPTH` (pad the prompt to this many tokens), `SDB_OUT` (per-token CSV).
- `run-sdb.sh` logs `tools/gpu-pstate/gpu-pstate` at 250 ms alongside (build it first, see its header) and prints a
  `PSTATE` summary for the timed window; `watch.sh` kills the run if free memory < 8% or swap grows > 1.2 GB.
- With `GGML_METAL_TIMELINE=<file>` and `SDB_MARKS=<file>`, `tl.py MARKS TIMELINE` breaks each decode into CPU, launch,
  GPU busy and readback.

## End to end: `bench3.py`

Against a running `SPLIT_DECODE_L=20 LLAMA_SPLIT_VERBOSE=1 scripts/serve.sh 2> serve.log`:

```bash
PHONE_IP=169.254.x.y python3 tools/neo-air/bench3.py serve.log decode_short decode_4k prefill_2k
```

- One JSON line per request: tok/s, Mac / phone / link ms per token (from the split-verbose lines), thermal before / after /
  max, memory on both sides.
- Before every request it waits for **phone thermal <= 1 and Mac thermal 0** (`bin/therm`), and at least 120 s.
- `decode_4k`: prefill 4k with `n_predict 1`, cool down, then append the output token + a user turn with `cache_prompt`
  and decode 128. The append must log no reset.
- Also `prefill_8k`, `rewire` (first request after > 90 s idle, then again at once).

## Small helpers

| tool | what |
|---|---|
| `bin/therm` | prints `NSProcessInfo.thermalState` (0 = nominal) |
| `bin/gpuwarm US SECONDS` | a 32-thread Metal dispatch every US µs, from a separate process (the side-process warm A/B) |
| `snap.sh LABEL [PID]` | Mac swap / pressure / wired, a process's footprint, and the phone's `mem` reply (`PHONE_IP`) |
| `watch.sh PID LOG` | memory watchdog used by the run scripts |
