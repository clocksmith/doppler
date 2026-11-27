#!/bin/bash
# Committed copy: the host-specific GPU lock path and gate directory are written as $GPU_LOCK and $GATE.
# DopplerX AMD gate: run one GPU step under the preconditions and a watchdog.
# - preconditions (precheck.sh) must pass right before the step;
# - the step holds the fleet GPU lock SHARED (as the standing Flash-Next participant does), so it
#   cannot start while a timing run holds it exclusively;
# - watchdog every 2 s: process-tree RSS, MemAvailable, memory pressure, GPU VRAM/GTT, task state.
#   One SIGTERM if tree RSS > 10 GiB, MemAvailable < 30 GiB or memory pressure avg10 > 5.
#   A task in D state for 60 s stops the watchdog: no kill, evidence is captured for the owner.
# usage: run-guarded.sh <label> <outdir> <timeout-s> -- <command...>
set -u
label=$1; outdir=$2; tmo=$3; shift 4
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$outdir"
log="$outdir/$label.guard.log"
"$here/precheck.sh" "$label" > "$outdir/$label.precheck.log" 2>&1
pc=$?
cat "$outdir/$label.precheck.log"
if [ $pc != 0 ]; then echo "NOT RUN: precheck failed" | tee -a "$log"; exit 3; fi
dev=/sys/class/drm/card1/device
start=$(date -u +%FT%TZ); startk=$(date '+%F %T')
echo "start $start: $*" > "$log"
flock --shared --nonblock $GPU_LOCK \
  nice -n 10 timeout -s TERM "$tmo" "$@" > "$outdir/$label.stdout.log" 2> "$outdir/$label.stderr.log" &
top=$!
peak=0; termed=0; dsec=0; dstuck=0
tree() { local p=$1; echo $p; for c in $(pgrep -P $p); do tree $c; done; }
while kill -0 $top 2>/dev/null; do
  pids=$(tree $top)
  rss=0; states=""
  for p in $pids; do
    r=$(awk '/^VmRSS/ {print $2}' /proc/$p/status 2>/dev/null); rss=$((rss + ${r:-0}))
    s=$(awk '/^State/ {print $2}' /proc/$p/status 2>/dev/null); states="$states$s"
  done
  [ $rss -gt $peak ] && peak=$rss
  av=$(awk '/^MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  mp=$(awk '/^some/ {sub("avg10=","",$2); print $2}' /proc/pressure/memory)
  vram=$(( $(cat $dev/mem_info_vram_used 2>/dev/null || echo 0) / 1048576 ))
  gtt=$(( $(cat $dev/mem_info_gtt_used 2>/dev/null || echo 0) / 1048576 ))
  echo "$(date -u +%T) rssMiB=$((rss/1024)) availGiB=$av memPsi10=$mp vramMiB=$vram gttMiB=$gtt states=$states" >> "$log"
  if [ $termed = 0 ] && { [ $rss -gt 10485760 ] || [ $av -lt 30 ] || awk -v m="$mp" 'BEGIN {exit !(m > 5)}'; }; then
    echo "GUARD: SIGTERM (rss=${rss}KiB avail=${av}GiB psi10=$mp)" | tee -a "$log"; kill -TERM $top; pkill -TERM -P $top; termed=1
  fi
  case "$states" in *D*) dsec=$((dsec + 2));; *) dsec=0;; esac
  if [ $dsec -ge 60 ]; then
    dstuck=1
    echo "D-STATE for 60 s: stopping the watchdog, no kill" | tee -a "$log"
    for p in $pids; do
      echo "--- pid $p" >> "$log"; grep -E "^(Name|State|SigPnd|ShdPnd|SigBlk)" /proc/$p/status >> "$log" 2>&1
      echo "wchan: $(cat /proc/$p/wchan 2>/dev/null)" >> "$log"; cat /proc/$p/stack >> "$log" 2>&1
    done
    journalctl -k --since "$startk" --no-pager >> "$log" 2>&1
    break
  fi
  sleep 2
done
if [ $dstuck = 1 ]; then echo "STUCK $label: see $log"; exit 4; fi
wait $top; rc=$?
end=$(date -u +%FT%TZ)
{
  echo "end $end rc=$rc peakTreeRssMiB=$((peak/1024)) sigterm=$termed"
  echo "kernel messages (amdgpu/drm) during the step:"
  journalctl -k --since "$startk" --no-pager -o short-iso 2>/dev/null | grep -i -E "amdgpu|drm|gpu" || echo "(none)"
  echo "flash-next partition-state events during the step:"
  journalctl --user -u reploid-flash-next.service --since "$startk" --no-pager -o cat 2>/dev/null | grep '"event":"partition-state"' | grep -o '"at":"[^"]*","phase":"[a-z]*"' || echo "(none)"
  echo "processes left from this step: $(pgrep -f "$outdir" | grep -v $$ | wc -l)"
} >> "$log"
tail -8 "$log"
exit $rc
