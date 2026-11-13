#!/bin/sh
# Phase 3 of run-fresh.sh `gpu` (frozen at b826337d): fixtures, plain run, routing controls, comparisons.
# Deviation: fixtures use tooling-patch/prepare-routing-fixture.py (tie-aware top-k consistency check,
# see deviations.json); every other command is verbatim from run-fresh.sh; precheck() is a host-safety addition.
set -eu
: "${SNAPSHOT:?}" "${ROOT:?}" "${MODEL_DIR:?}" "${SOURCE_DIR:?}" "${GPU_LOCK:?}"
F="$SNAPSHOT/reports/flash-next-routing-aware-fresh-20261112"
PROMPTS="$SNAPSHOT/tools/data/flash-next-routing-acceptance-prompts-v1.json"
INPUTS="$ROOT/frozen/inputs.json"
CASES=$(python3 -c "import json;print(' '.join(c['id'] for c in json.load(open('$PROMPTS'))['cases']))")
stamp() { printf '{"run":"%s","startedAtUtc":"%s"}\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" >> "$ROOT/run-starts.jsonl"; }
gpu() { name=$1; script=$2; shift 2; stamp "$name"; (cd "$SNAPSHOT" && env "$@" flock --shared "$GPU_LOCK" nice -n 10 node "$script" > "$ROOT/gpu/$name.log" 2>&1); }
precheck() {
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  swap=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo)
  bova=$(journalctl -k -b --no-pager 2>/dev/null | grep -c 'BO_VA' || true)
  dstate=$(ps -eo stat=,wchan:40=,pid=,comm= | awk '$1 ~ /^D/ && $2 ~ /amdgpu|drm|ttm|dma_fence/' | wc -l)
  echo "$(date -u +%FT%TZ) precheck $1 availGiB=$avail swapUsedMiB=$swap bova=$bova dstateAmdgpu=$dstate" >> "$ROOT/gpu-precheck.log"
  if [ "$avail" -lt 30 ] || [ "$swap" -gt 1024 ] || [ "$bova" -gt 0 ] || [ "$dstate" -gt 0 ]; then
    echo "$(date -u +%FT%TZ) STOP before $1" >> "$ROOT/gpu-precheck.log"; exit 3
  fi
}

mkdir -p "$ROOT/gpu/plain/all" "$ROOT/gpu/observe" "$ROOT/gpu/indices" "$ROOT/gpu/forced-comparison" "$ROOT/fixture"
REF="$ROOT/cpu/all/round-0/learned-forward-reference-float32-artifact.json"
[ -f "$ROOT/cpu/all/round-0/round.json" ] || { echo "CPU round 0 missing; stopping"; exit 4; }
for id in $CASES; do
  python3 -P "$ROOT/tooling-patch/prepare-routing-fixture.py" "$ROOT/cpu/all/round-0/router-$id" \
    "$ROOT/fixture/$id" "$MODEL_DIR/manifest.json" "$INPUTS" "$REF" "$id"
done
precheck plain
gpu plain tests/integration/flash-next-learned-physical.test.js \
  DOPPLER_FLASH_NEXT_PROBE="$ROOT/config/plain-all.json" || echo "plain run failed"
precheck routing-controls
stamp routing-controls
(cd "$SNAPSHOT" && flock --shared "$GPU_LOCK" nice -n 10 node "$F/run-routing-controls.js" \
  "$ROOT/config/routing-controls-all.json" > "$ROOT/gpu/routing-controls.log" 2>&1) || echo "routing controls failed"
for id in $CASES; do
  python3 -P "$SNAPSHOT/reports/flash-next-forced-routing-20261112/compare-forced-routing.py" \
    "$ROOT/gpu/forced-comparison/$id.json" "$REF" "$SNAPSHOT/src/config/evidence/boundary-evidence-policy.json" "$INPUTS" "$id" \
    "indices=$ROOT/gpu/indices/$id/observation.json" || echo "comparison $id failed"
done
echo "gpu-track-part3 done $(date -u +%FT%TZ)"
