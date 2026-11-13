#!/bin/sh
# Same thresholds as pressure-guard.sh (PSI some avg10 > 2 or MemAvailable < 20 GiB); bracketed pgrep pattern.
R=/home/x/flash-next-routing-aware-fresh-20261112
sleep 30
while pgrep -f "[c]pu-reference.py" > /dev/null; do
  p=$(awk "/^some/{split(\$2,a,\"=\"); print a[2]}" /proc/pressure/memory)
  avail=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)
  swap=$(awk "/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print int((t-f)/1024)}" /proc/meminfo)
  echo "$(date -u +%FT%TZ) some_avg10=$p availGiB=$avail swapUsedMiB=$swap" >> $R/pressure.log
  if [ "$(echo "$p > 2" | bc)" = 1 ] || [ "$avail" -lt 20 ]; then
    echo "$(date -u +%FT%TZ) STOP: pressure $p avail $avail" >> $R/pressure.log
    pkill -f "[f]lash-next-routing-aware-fresh-20261112/cpu-reference.py"; exit 1
  fi
  sleep 20
done
