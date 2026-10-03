#!/usr/bin/env python3
"""turn-bench.py - how fast a new tool result is read at the depths agent sessions sit at, Mac alone vs Mac + iPhone.

Agent sessions grow turn by turn with the prompt cache; each turn reads only its new tokens. So the context is built once and
saved (--build: one read, slots saved at each --depths), and every measurement restores a saved depth in seconds and then
reads --reps new tool results of --chunk tokens (llama.cpp's server sources, which the context doesn't contain). The first
read after a restore includes what real use pays after a cache restore (with the phone: its copy of the context is sent once).

  scripts/turn-bench.py --build                      # once: read to the deepest depth, save a slot at each depth (Mac alone)
  scripts/turn-bench.py --config mac                 # the Mac alone (same server settings, no phone)
  scripts/turn-bench.py --config phone               # Mac + iPhone
  ENV=VALUE ... scripts/turn-bench.py --config phone --note "what changed"   # an A/B: same depths, same reads
Rows: bench/results/turns.jsonl (one per read: depth, tokens, ms, tok/s, config, note, the server's env knobs).
"""
import argparse, glob, json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--build', action='store_true')
ap.add_argument('--config', choices=['mac', 'phone'], default='phone')
ap.add_argument('--depths', default='16384,32768,49152')
ap.add_argument('--chunk', type=int, default=2048, help='tokens per tool result')
ap.add_argument('--reps', type=int, default=2, help='tool results read per depth')
ap.add_argument('--port', type=int, default=8093)
ap.add_argument('--note', default='')
a = ap.parse_args()
URL = f'http://127.0.0.1:{a.port}'
OUT = os.path.join(ROOT, 'bench', 'results')
SLOTS = os.path.join(OUT, 'slots', 'turns')
os.makedirs(SLOTS, exist_ok=True)
depths = sorted(int(x) for x in a.depths.split(','))
say = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)
KNOBS = ['SPLIT_UB', 'LLAMA_SPLIT_MIN', 'MM_SME', 'CTX_CHECKPOINTS', 'LLAMA_UBATCH_REMOTE', 'SME', 'LOAD_MODE']


def post(path, body, timeout=3600):
    r = urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'}),
                               timeout=timeout)
    return json.loads(r.read())


def swap_gb():
    m = re.search(r'used = ([\d.]+)M', subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout)
    return round(float(m.group(1)) / 1024, 2) if m else None


def record(**kw):
    with open(os.path.join(OUT, 'turns.jsonl'), 'a') as f:
        f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), note=a.note, swap_gb=swap_gb(),
                                knobs={k: os.environ[k] for k in KNOBS if k in os.environ}, **kw)) + '\n')


cfg = 'mac' if a.build else a.config
log_path = os.path.join(OUT, f'turn-server-{cfg}.log')
env = dict(os.environ, PROXY='0', PORT=str(a.port), CACHE_DIR=SLOTS)
env.setdefault('CTX', '65536')   # the caller's CTX / KV win (e.g. a 16k config)
env.setdefault('KV', 'q8_0')
if cfg == 'mac':
    env['PHONE'] = '0'
srv = subprocess.Popen([f'{ROOT}/scripts/serve.sh'], cwd=ROOT, env=env, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            if urllib.request.urlopen(URL + '/health', timeout=2).status == 200:
                break
        except Exception:
            pass
        if srv.poll() is not None:
            sys.exit(f'server exited, see {log_path}')
        time.sleep(1)
    line = next((l.strip() for l in open(log_path) if 'serve:' in l), '')
    if cfg == 'phone' and 'split prefill on' not in line:
        sys.exit(f'the phone is not in use: {line!r}')
    say(f'{cfg} server up {line}')
    tok = lambda s: post('/tokenize', {'content': s})['tokens']
    base = tok('<|im_start|>user\n' + ''.join(open(f, errors='replace').read() for f in sorted(glob.glob(f'{ROOT}/llama.cpp/src/*.cpp'))))
    if len(base) < depths[-1]:
        sys.exit(f'only {len(base)} tokens of text')
    if a.build:
        n = 0
        for d in depths:
            t0 = time.time()
            r = post('/completion', {'prompt': base[:d], 'n_predict': 0, 'cache_prompt': True, 'temperature': 0})
            say(f'  read to {d}: {r["timings"]["prompt_n"]} tokens in {time.time() - t0:.0f} s')
            post('/slots/0?action=save', {'filename': f'turns-{d}.bin'})
            n = d
        say('built', ', '.join(f'turns-{d}.bin' for d in depths))
        sys.exit(0)
    tools = tok(''.join(open(f, errors='replace').read() for f in sorted(glob.glob(f'{ROOT}/llama.cpp/tools/server/*.cpp'))))
    for d in depths:
        t0 = time.time()
        rs = post('/slots/0?action=restore', {'filename': f'turns-{d}.bin'})
        restore_s = time.time() - t0
        seq = base[:d]
        for k in range(a.reps):
            chunk = tools[k * a.chunk:(k + 1) * a.chunk]
            t0 = time.time()
            r = post('/completion', {'prompt': seq + chunk, 'n_predict': 0, 'cache_prompt': True, 'temperature': 0})
            wall = time.time() - t0
            tm = r['timings']
            say(f'  depth {d:6d} read {k + 1}: {tm["prompt_n"]} tokens in {tm["prompt_ms"] / 1000:.2f} s = '
                f'{tm["prompt_n"] / (tm["prompt_ms"] / 1000):.1f} tok/s (wall {wall:.2f} s)')
            record(kind='turn_read', config=cfg, depth=d, read=k + 1, prompt_n=tm['prompt_n'], prompt_ms=tm['prompt_ms'],
                   tok_s=tm['prompt_n'] / (tm['prompt_ms'] / 1000), wall_s=wall, restore_s=restore_s if k == 0 else None,
                   restored=rs.get('n_restored'))
            seq = seq + chunk
finally:
    srv.terminate()
    try:
        srv.wait(30)
    except subprocess.TimeoutExpired:
        srv.kill()
    subprocess.run(['pkill', '-f', f'llama-server .*--port {a.port}'])
