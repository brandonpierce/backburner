#!/bin/bash
# watch.sh PID LOG: every 5 s log swap / free% / wired; kill PID (TERM) if free% < 8 or swap grows > 1200 MB from the start
sw0=$(sysctl -n vm.swapusage | awk '{print int($6)}')
while kill -0 $1 2>/dev/null; do
  f=$(memory_pressure -Q | awk -F': ' '/percentage/{print $2+0}'); sw=$(sysctl -n vm.swapusage | awk '{print int($6)}')
  w=$(vm_stat | awk '/wired down/{print int($4*16/1024)}')
  echo "$(date +%T) free=${f}% swap=${sw}M (+$((sw-sw0))) wired=${w}MiB" >> $2
  if [ "$f" -lt 8 ] || [ $((sw - sw0)) -gt 1200 ]; then echo "$(date +%T) WATCHDOG KILL" >> $2; kill -TERM $1; sleep 5; kill -KILL $1 2>/dev/null; fi
  sleep 5
done
