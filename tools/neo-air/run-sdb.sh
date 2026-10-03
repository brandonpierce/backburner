#!/bin/bash
# run-sdb.sh TAG [extra args]: one sdbench run with serve.sh SPLIT_DECODE_L=20 flags + env; gpu-pstate alongside; watchdog
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$D/out}; R=$(cd "$D/../.." && pwd)
TAG=$1; shift
O=$OUT/sdb_$TAG; mkdir -p "$OUT"
MODEL=${MODEL:-$HOME/Models/Qwen3.8-27B-IQ2_XS.gguf}
T=${T:-2}; TB=${TB:-$T}
L=${LLAMA_SPLIT_L:-20}
echo "== $TAG $(date +%T) swap: $(sysctl -n vm.swapusage | awk '{print $6}') free: $(memory_pressure -Q | awk -F': ' '/percentage/{print $2}')" | tee $O.txt
"$R/tools/gpu-pstate/gpu-pstate" 250 > $O.pstate & GP=$!
env LLAMA_LAZY_EMBD=${LLAMA_LAZY_EMBD:-1} GGML_METAL_REGFED=${GGML_METAL_REGFED:-1} GGML_METAL_FA_GQA=${GGML_METAL_FA_GQA:-1} \
  GGML_METAL_FA_PREFILL_GQA=${GGML_METAL_FA_PREFILL_GQA:-1} LLAMA_BATCHED_ARGMAX=1 SPEC_DRAFT_UBATCH=64 LLAMA_SPLIT_L=$L \
  SDB_OUT=$O.csv SDB_MARKS=$O.marks \
  "$D/bin/sdbench" -m $MODEL -ngl 999 -fa on -c ${CTX:-16384} -np 1 -ctk q8_0 -ctv q8_0 -t $T -tb $TB --spec-type none \
  --ctx-checkpoints 0 --cache-ram 0 -ub 256 \
  -ot "^blk\.(${GE:-20|[1-9][0-9][0-9]+|[3-9][0-9]|2[1-9]})\.=CPU" -ot '^output=CPU' --no-repack "$@" >> $O.txt 2> $O.err & PID=$!
"$D/watch.sh" $PID $O.watch &
wait $PID; RC=$?
kill $GP 2>/dev/null
W=$(awk '/^WINDOW/{print $2, $3}' $O.txt)
python3 - "$O.pstate" $W >> $O.txt <<'PY'
import sys
f, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = []
for l in open(f):
    p = l.split()
    t = int(p[0])
    if a + 1 <= t <= b - 1:
        rows.append((float(p[1].rstrip('%')), float(p[2]), float(p[5].rstrip('%')), float(p[6])))
if rows:
    n = len(rows); m = lambda i: sum(r[i] for r in rows) / n
    print(f"PSTATE n={n} active {m(0):.1f}% avgMHz {m(1):.0f} top {m(2):.1f}% W {m(3):.2f}")
PY
echo "rc=$RC" >> $O.txt
grep -E "RESULT|PSTATE|rc=" $O.txt; tail -1 $O.watch
