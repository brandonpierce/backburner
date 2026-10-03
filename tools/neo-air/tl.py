# tl.py MARKS TIMELINE: per-decode breakdown from sdbench marks (D ta tb) + GGML_METAL_TIMELINE (G/C/W lines), mach clock
import sys, statistics as st
D = [tuple(map(float, l.split()[1:3])) for l in open(sys.argv[1]) if l.startswith('D')]
G, C, W = [], [], []
for l in open(sys.argv[2]):
    p = l.split()
    if p[0] == 'G': G.append(float(p[3]))
    elif p[0] == 'C' and float(p[4]) > 0: C.append((p[2], float(p[4]), float(p[5])))
    elif p[0] == 'W': W.append((float(p[2]), float(p[3])))
rows = []
for ta, tb in D:
    g = [x for x in G if ta <= x <= tb]
    c = sorted([x for x in C if ta <= x[1] <= tb], key=lambda x: x[1])
    w = [x for x in W if ta <= x[0] <= tb]
    if not g or not c: continue
    busy, end = 0.0, 0.0
    for k, a, b in c:
        a = max(a, end); busy += max(0, b - a); end = max(end, b)
    cg = [x for x in c if x[0] == 'g']
    rows.append(dict(wall=(tb - ta), pre=g[0] - ta, launch=(cg[0][1] if cg else c[0][1]) - g[0], gpu_busy=busy,
                     gpu_span=c[-1][2] - c[0][1], n_cb=len(c), n_g=len(g), wait=sum(b - a for a, b in w),
                     post=tb - c[-1][2]))
if not rows: sys.exit('no rows')
print(f"TL n={len(rows)} cmdbufs/decode {st.mean(r['n_cb'] for r in rows):.1f} graph_computes/decode {st.mean(r['n_g'] for r in rows):.1f}")
for k in ('wall', 'pre', 'launch', 'gpu_busy', 'gpu_span', 'wait', 'post'):
    v = sorted(r[k] * 1e3 for r in rows)
    print(f"TL {k:9s} mean {st.mean(v):7.2f}  p50 {v[len(v)//2]:7.2f}  p95 {v[int(len(v)*.95)]:7.2f} ms")
