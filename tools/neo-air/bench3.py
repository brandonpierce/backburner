# bench3.py LOG WHICH...: end-to-end numbers against a running split-decode serve.sh (port 8080; LOG is its log, run with
#   LLAMA_SPLIT_VERBOSE=1). One JSON line per run.
#   decode_short (3x128), decode_4k (3x: prefill 4k, cool to thermal <= 1, then a true append + 128 decoded), prefill_2k (3x cold),
#   prefill_8k (1x), rewire (decode after a > 90 s idle pause, then again at once)
import json, os, re, subprocess, sys, threading, time, urllib.request

IP = os.environ['PHONE_IP']   # the phone's 169.254.x.y address (scripts/phone-up.sh prints it)
URL = 'http://127.0.0.1:8080'
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
LOG = sys.argv[1]


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def phone():
    try:
        out = subprocess.run(['nc', '-G', '2', '-w', '3', IP, '50061'], input=b'mem\n', capture_output=True, timeout=10).stdout
        d = json.loads(out)
        return {k: round(d[k], 1) if isinstance(d[k], float) else d[k] for k in ('thermal', 'sys_wired_mb', 'avail_mb', 'footprint_mb')}
    except Exception as e:
        return {'error': str(e)}


def mac():
    sw = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout.split()
    vm = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    wired = next(int(l.split()[-1].rstrip('.')) for l in vm.splitlines() if 'wired down' in l) * 16384 / 2**20
    pid = subprocess.run(['pgrep', '-f', 'bin/llama-server'], capture_output=True, text=True).stdout.split()[0]
    fp = subprocess.run(['footprint', '-p', pid], capture_output=True, text=True).stdout
    fp_mb = next((l.split('Footprint:')[1].strip() for l in fp.splitlines() if 'Footprint:' in l), '?')
    free = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True).stdout
    return {'swap_used_mb': float(sw[5].rstrip('M')), 'wired_mib': round(wired), 'server_footprint': fp_mb,
            'free_pct': int(re.search(r'(\d+)%', free).group(1))}


def corpus_tokens():
    txt = ''
    for p in ['README.md', 'docs/ANE.md', 'docs/INSTALL-IPHONE.md', 'docs/TWO-PHONES.md', 'scripts/serve.sh',
              'scripts/proxy.py', 'scripts/split-gguf.py', 'scripts/phone-up.sh']:
        txt += f'\n\n===== {p} =====\n' + open(f'{ROOT}/{p}').read()
    return post('/tokenize', {'content': txt})['tokens']


def mac_thermal():
    return int(subprocess.run([os.path.join(HERE, 'bin', 'therm')], capture_output=True, text=True).stdout.strip() or 9)


def cool(limit_s=1200, min_s=120):
    # phone thermal <= 1 AND the (fanless) Mac back at nominal, and at least min_s since the last request
    t0 = time.time()
    while True:
        th, tm = phone().get('thermal', 9), mac_thermal()
        if (th <= 1 and tm == 0 and time.time() - t0 >= min_s) or time.time() - t0 > limit_s:
            return th, round(time.time() - t0)
        time.sleep(15)


def logpos():
    return open(LOG, 'rb').seek(0, 2)


def logslice(a):
    with open(LOG, 'rb') as f:
        f.seek(a)
        return f.read().decode(errors='replace')


def split_stats(txt):
    rows = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in
            re.finditer(r'split decode 1 tokens at \d+: Mac ([\d.]+) ms, wait ([\d.]+) ms \(phone ([\d.]+) ms', txt)]
    if not rows:
        return None
    n = len(rows); s = sorted(rows)
    mean = lambda i: round(sum(r[i] for r in rows) / n, 1)
    p50 = lambda i: round(sorted(r[i] for r in rows)[n // 2], 1)
    return {'n': n, 'mac_ms': mean(0), 'mac_p50': p50(0), 'wait_ms': mean(1), 'phone_ms': mean(2), 'phone_p50': p50(2),
            'link_ms': round(mean(1) - mean(2), 1)}


def measured(name, body, extra=None):
    """one request with thermal polling; returns the record"""
    p0, m0, a = phone(), mac(), logpos()
    mt0 = mac_thermal()
    seen, stop = [p0.get('thermal')], threading.Event()
    def poll():
        while not stop.wait(5):
            seen.append(phone().get('thermal'))
    th = threading.Thread(target=poll); th.start()
    t0 = time.time()
    r = post('/completion', body)
    wall = time.time() - t0
    stop.set(); th.join()
    p1, m1 = phone(), mac()
    txt = logslice(a)
    t = r['timings']
    tmax = max(x for x in seen + [p1.get('thermal')] if x is not None)
    rec = {'run': name, 'prompt_n': t['prompt_n'], 'prompt_ms': round(t['prompt_ms']), 'pp_tok_s': round(t['prompt_per_second'], 2),
           'predicted_n': t['predicted_n'], 'tg_tok_s': round(t['predicted_per_second'], 3), 'wall_s': round(wall, 1),
           'thermal': [p0.get('thermal'), p1.get('thermal')], 'thermal_max': tmax, 'mac_thermal': [mt0, mac_thermal()],
           'label': 'clean' if tmax <= 1 else f'throttled (thermal {tmax})',
           'reset_lines': len(re.findall(r'resetting the worker|seq_rm at position|re-prefill', txt)),
           'split': split_stats(txt), 'phone_after': p1, 'mac_after': m1,
           'swap_delta_mb': round(m1['swap_used_mb'] - m0['swap_used_mb'], 1)}
    if extra: rec.update(extra)
    print(json.dumps(rec), flush=True)
    if tmax >= 3:
        print(json.dumps({'STOP': 'phone reached thermal 3'}), flush=True); sys.exit(3)
    return rec, r


if __name__ == '__main__':
    toks = corpus_tokens()
    print(json.dumps({'corpus_tokens': len(toks)}), flush=True)
    short = 'The history of the printing press began'
    for w in sys.argv[2:]:
        if w == 'decode_short':
            for k in range(3):
                th, waited = cool()
                measured(f'decode_short#{k+1}', {'prompt': short, 'n_predict': 128, 'temperature': 0, 'cache_prompt': False, 'ignore_eos': True},
                         {'cooldown_s': waited})
        elif w == 'decode_4k':
            user = post('/tokenize', {'content': '\n\nQuestion: summarize the documents above in five short bullet points.\nAnswer:'})['tokens']
            for k in range(3):
                P = toks[k * 7:k * 7 + 4096]   # shifted per run: run k+1 is a cold prefill, not a cache hit
                th, waited = cool()
                rec1, r1 = measured(f'decode_4k#{k+1}/prefill', {'prompt': P, 'n_predict': 1, 'temperature': 0, 'cache_prompt': True,
                                                                  'return_tokens': True}, {'cooldown_s': waited})
                out = r1.get('tokens') or []
                th, waited = cool()
                measured(f'decode_4k#{k+1}/append', {'prompt': P + out + user, 'n_predict': 128, 'temperature': 0, 'cache_prompt': True,
                                                     'ignore_eos': True}, {'cooldown_s': waited, 'appended_tokens': len(out) + len(user)})
        elif w == 'prefill_2k':
            for k in range(3):
                th, waited = cool()
                measured(f'prefill_2k#{k+1}', {'prompt': toks[k * 7:k * 7 + 2048], 'n_predict': 1, 'temperature': 0, 'cache_prompt': False},
                         {'cooldown_s': waited})
        elif w == 'prefill_8k':
            th, waited = cool()
            measured('prefill_8k', {'prompt': toks[:8192], 'n_predict': 1, 'temperature': 0, 'cache_prompt': False}, {'cooldown_s': waited})
        elif w == 'rewire':
            # first: after > 90 s without any graph on the phone (keep-alive 60 s: the tail unwires), then immediately again
            th, waited = cool()
            time.sleep(max(0, 95 - waited))
            print(json.dumps({'phone_before_rewire': phone()}), flush=True)
            measured('rewire/after_95s_idle', {'prompt': short, 'n_predict': 32, 'temperature': 0, 'cache_prompt': False, 'ignore_eos': True})
            measured('rewire/immediately', {'prompt': short, 'n_predict': 32, 'temperature': 0, 'cache_prompt': False, 'ignore_eos': True})
