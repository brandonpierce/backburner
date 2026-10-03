#!/bin/bash
# snap.sh LABEL [PID]: Mac swap / pressure / wired + server footprint + phone mem, one block
IP=${PHONE_IP:?set PHONE_IP to the phone's 169.254.x.y address}
echo "== $1 $(date +%T)"
echo "mac swap: $(sysctl -n vm.swapusage)"
echo "mac pressure: $(memory_pressure -Q 2>/dev/null | tail -1)"
vm_stat | awk '/wired down|compressor|Pages free/{printf "mac %s %s  ", $0, ""} END{print ""}' | tr -s ' '
if [ -n "${2:-}" ]; then
  echo "server rss_kb: $(ps -o rss= -p $2)"
  footprint -p $2 2>/dev/null | grep -E 'Footprint|Untagged|IOKit|Dirty' | head -5
fi
printf 'mem\n' | nc -G 2 -w 3 $IP 50061 | python3 -c 'import sys,json
try:
  d=json.loads(sys.stdin.read()); print("phone:", {k:d.get(k) for k in ("tail_state","sys_wired_mb","avail_mb","footprint_mb","sys_free_mb","thermal","tail_sessions")})
except Exception as e: print("phone: no answer", e)'
