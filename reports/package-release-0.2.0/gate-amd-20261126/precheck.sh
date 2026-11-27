#!/bin/bash
# Committed copy: the host-specific GPU lock path and gate directory are written as $GPU_LOCK and $GATE.
# DopplerX AMD gate: preconditions on the AMD host, run immediately before each GPU step.
# Prints a report and exits 0 only if every precondition holds. Reads /proc and /sys only; no GPU work.
# usage: precheck.sh <label>
label=${1:-check}
fail=0
note() { echo "$1"; }
bad() { echo "FAIL $1"; fail=1; }
echo "== precheck $label at $(date -u +%FT%TZ)"
echo "load: $(cat /proc/loadavg)"

# 1. Disk I/O: sample /proc/diskstats over 10 s for the physical disks.
snap() { awk '$3 ~ /^(sd[a-z]|nvme[0-9]+n[0-9]+)$/ {r+=$6; w+=$10} END {print r, w}' /proc/diskstats; }
read r1 w1 < <(snap); sleep 10; read r2 w2 < <(snap)
rmb=$(( (r2 - r1) * 512 / 10 / 1048576 )); wmb=$(( (w2 - w1) * 512 / 10 / 1048576 ))
note "disk: read ${rmb} MiB/s, write ${wmb} MiB/s (10 s mean)"
[ "$rmb" -gt 20 ] && bad "disk read ${rmb} MiB/s > 20"
[ "$wmb" -gt 20 ] && bad "disk write ${wmb} MiB/s > 20"
heavy=$(ps -eo pid,etime,args | grep -E "(^|/| )(rsync|scp|dd|rclone|tar|find|cp|sha256sum|b3sum|huggingface-cli) " | grep -v -E "grep|precheck" | cut -c1-160)
if [ -n "$heavy" ]; then bad "copy/scan process running:"; echo "$heavy"; else note "no rsync/scp/dd/find/tar/cp/hash process"; fi
note "io pressure: $(head -1 /proc/pressure/io)"

# 2. Memory: pressure zero, MemAvailable stable over a minute (7 samples, 10 s apart; the first 10 s were above).
mp_sys=$(awk '/^some/ {print $2, $3}' /proc/pressure/memory)
mp_user=$(awk '/^some/ {print $2, $3}' /sys/fs/cgroup/user.slice/memory.pressure)
note "memory pressure: system [$mp_sys] user.slice [$mp_user]"
for v in $mp_sys $mp_user; do case "$v" in avg10=0.00|avg60=0.00) ;; *) bad "memory pressure nonzero ($v)";; esac; done
avs=()
for i in 1 2 3 4 5 6 7; do avs+=($(awk '/^MemAvailable/ {print int($2/1048576)}' /proc/meminfo)); [ $i -lt 7 ] && sleep 10; done
min=${avs[0]}; max=${avs[0]}; for a in "${avs[@]}"; do [ $a -lt $min ] && min=$a; [ $a -gt $max ] && max=$a; done
note "MemAvailable GiB over 60 s: ${avs[*]}"
[ $((max - min)) -gt 3 ] && bad "MemAvailable moved $((max - min)) GiB in 60 s"
[ $min -lt 40 ] && bad "MemAvailable ${min} GiB < 40"
note "swap: $(awk '/^SwapTotal|^SwapFree/ {printf "%s %d MiB ", $1, $2/1024}' /proc/meminfo)"
mp_sys2=$(awk '/^some/ {print $2}' /proc/pressure/memory); [ "$mp_sys2" != "avg10=0.00" ] && bad "memory pressure rose during the sample ($mp_sys2)"

# 3. GPU holders: only the desktop and the standing Flash-Next participant may hold the device.
flash_pid=$(systemctl --user show reploid-flash-next.service -p MainPID --value)
flash_node=$(pgrep -P "$(pgrep -P "$flash_pid" | head -1)" | head -1)
holders=$(for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f" 2>/dev/null); case "$t" in /dev/dri/*|/dev/kfd) p=${f#/proc/}; echo "${p%%/*}";; esac; done 2>/dev/null | sort -u)
for p in $holders; do
  c=$(cat /proc/$p/comm 2>/dev/null)
  case "$c" in gnome-shell|Xwayland|mutter-x11-fram|ibus-x11) ;;
    *) if [ "$p" = "$flash_node" ]; then :; else bad "GPU held by pid $p ($c): $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-140)"; fi;;
  esac
done
note "GPU holders: $(for p in $holders; do printf '%s(%s) ' $p "$(cat /proc/$p/comm 2>/dev/null)"; done)"
note "flash-next: unit MainPID $flash_pid, node pid $flash_node"
if flock --shared --nonblock $GPU_LOCK true; then note "doe-gpu-timing.lock: no exclusive holder"; else bad "doe-gpu-timing.lock held exclusively"; fi
dstate=$(ps -eo pid,stat,comm,wchan:32 | awk '$2 ~ /^D/' | grep -v -E "kworker|jbd2|kcryptd|dmcrypt" )
[ -n "$dstate" ] && { bad "process in D state:"; echo "$dstate"; }

# 4. Other lanes: no Fawn or other GPU/timing job active (standing idle services are listed, not failed).
others=$(ps -eo pid,etimes,pcpu,args --sort=-pcpu | grep -E "fawn|task-run|nbranch|value-run|chrome|chromium|playwright|webgpu|doppler|vulkan|doe-gpu" | grep -v -E "grep|precheck|dopplerx-amd-gate" | awk -v f="$flash_node" '$1 != f')
active=$(echo "$others" | awk '$1 != "" && ($3 > 2.0 || $2 < 3600)')
# Owner instruction 2026-11-27 ~00:05 UTC (via the coordinator): the Fawn rc2 run renders with SwiftShader
# on the CPU and never opens the AMD GPU, so an active other-lane browser job is recorded, not failed.
# Any other process holding /dev/dri or /dev/kfd still fails above.
if [ -n "$active" ]; then note "NOTE other lane process active (recorded; GPU use is checked by device holders):"; echo "$active" | cut -c1-180 | head -6; fi
kfd=$(for f in /proc/[0-9]*/fd/*; do [ "$(readlink "$f" 2>/dev/null)" = /dev/kfd ] && { p=${f#/proc/}; echo "${p%%/*}"; }; done 2>/dev/null | sort -u)
note "/dev/kfd holders: ${kfd:-none}"
[ -n "$kfd" ] && bad "/dev/kfd held by: $kfd"
note "standing other-lane processes (idle): $(echo "$others" | grep -c .)"

# 5. Flash-Next idle: the last partition-state event says ready, not executing.
last=$(journalctl --user -u reploid-flash-next.service --since "-3 h" --no-pager -o cat 2>/dev/null | grep '"event":"partition-state"' | tail -1 | grep -o '"phase":"[a-z]*"' | head -1)
lastat=$(journalctl --user -u reploid-flash-next.service --since "-3 h" --no-pager -o cat 2>/dev/null | grep '"event":"partition-state"' | tail -1 | grep -o '"at":"[^"]*"' | head -1)
note "flash-next last partition-state: ${last:-none in 3 h} ${lastat}"
case "$last" in ''|'"phase":"ready"') ;; *) bad "flash-next not idle ($last)";; esac
note "flash-next cpu: $(ps -o pcpu=,rss= -p "$flash_node")"

if [ $fail = 0 ]; then echo "PRECHECK PASS $label"; else echo "PRECHECK FAIL $label"; fi
exit $fail
