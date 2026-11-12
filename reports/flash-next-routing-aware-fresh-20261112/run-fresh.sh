#!/bin/sh
# Routing-aware fresh test, procedure steps 3-8, after the freeze record is committed and the
# inputs are frozen (freeze-inputs.py). Two tracks run concurrently:
#   cpu: cpu-reference.py chain (round 0 with router rows and sampled logits, then greedy rounds)
#   gpu: task quality, same-input capture, then (once CPU round 0 exists) fixtures, the plain run,
#        unforced observe and forced-indices runs per case, and their quantized-v1 comparisons.
# Every GPU job holds the shared GPU lock; CPU jobs are niced. Nothing is tuned between steps.
#
# Usage: run-fresh.sh cpu|gpu|audit|evaluate
# Environment: SNAPSHOT (runtime snapshot root), ROOT (run directory with frozen/ and plan.json),
#              MODEL_DIR, SOURCE_DIR (pinned checkpoint), GPU_LOCK (shared lock file),
#              CPU_THREADS (default 8).
set -eu
: "${SNAPSHOT:?}" "${ROOT:?}" "${MODEL_DIR:?}" "${SOURCE_DIR:?}" "${GPU_LOCK:?}"
F="$SNAPSHOT/reports/flash-next-routing-aware-fresh-20261112"
PRODUCER="$SNAPSHOT/artifacts/qwen38-flash-next-20261013/prepare-learned-forward-reference.py"
PROMPTS="$SNAPSHOT/tools/data/flash-next-routing-acceptance-prompts-v1.json"
INPUTS="$ROOT/frozen/inputs.json"
CASES=$(python3 -c "import json;print(' '.join(c['id'] for c in json.load(open('$PROMPTS'))['cases']))")
stamp() { printf '{"run":"%s","startedAtUtc":"%s"}\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" >> "$ROOT/run-starts.jsonl"; }
gpu() { name=$1; script=$2; shift 2; stamp "$name"; (cd "$SNAPSHOT" && env "$@" flock --shared "$GPU_LOCK" nice -n 10 node "$script" > "$ROOT/gpu/$name.log" 2>&1); }

case "$1" in
cpu)
  stamp cpu-chain
  OMP_NUM_THREADS="${CPU_THREADS:-8}" nice -n 10 python3 -P "$F/cpu-reference.py" chain "$PRODUCER" "$SOURCE_DIR" \
    "$MODEL_DIR" "$PROMPTS" "$INPUTS" "$ROOT/cpu/all" "$(echo $CASES | tr ' ' ',')"
  ;;
gpu)
  mkdir -p "$ROOT/gpu/plain/all" "$ROOT/gpu/observe" "$ROOT/gpu/indices" "$ROOT/gpu/forced-comparison" "$ROOT/same-input" "$ROOT/fixture"
  gpu task-quality tests/integration/flash-next-task-quality-physical.test.js \
    DOPPLER_FLASH_NEXT_TASK_QUALITY="$ROOT/config/task-quality-run.json" || echo "task-quality failed"
  gpu same-input-capture tests/integration/flash-next-same-input-capture-physical.test.js \
    DOPPLER_FLASH_NEXT_SAME_INPUT_CAPTURE="$ROOT/config/same-input-run.json" || echo "same-input capture failed"
  REF="$ROOT/cpu/all/round-0/learned-forward-reference-float32-artifact.json"
  until [ -f "$ROOT/cpu/all/round-0/round.json" ]; do sleep 60; done
  for id in $CASES; do
    python3 -P "$SNAPSHOT/reports/flash-next-forced-routing-20261112/prepare-routing-fixture.py" "$ROOT/cpu/all/round-0/router-$id" \
      "$ROOT/fixture/$id" "$MODEL_DIR/manifest.json" "$INPUTS" "$REF" "$id"
  done
  gpu plain tests/integration/flash-next-learned-physical.test.js \
    DOPPLER_FLASH_NEXT_PROBE="$ROOT/config/plain-all.json" || echo "plain run failed"
  stamp routing-controls
  (cd "$SNAPSHOT" && flock --shared "$GPU_LOCK" nice -n 10 node "$F/run-routing-controls.js" \
    "$ROOT/config/routing-controls-all.json" > "$ROOT/gpu/routing-controls.log" 2>&1) || echo "routing controls failed"
  for id in $CASES; do
    python3 -P "$SNAPSHOT/reports/flash-next-forced-routing-20261112/compare-forced-routing.py" \
      "$ROOT/gpu/forced-comparison/$id.json" "$REF" "$SNAPSHOT/src/config/evidence/boundary-evidence-policy.json" "$INPUTS" "$id" \
      "indices=$ROOT/gpu/indices/$id/observation.json" || echo "comparison $id failed"
  done
  ;;
audit)
  stamp same-input-audit
  AUDIT_THREADS="${CPU_THREADS:-8}" nice -n 10 python3 -P "$F/audit-same-input-all-layers.py" \
    --producer-dir "$SNAPSHOT/artifacts/qwen38-flash-next-20261013" --source-dir "$SOURCE_DIR" --artifact-dir "$MODEL_DIR" \
    --capture-dir "$ROOT/same-input/capture" --inputs "$INPUTS" --out "$ROOT/same-input/audit.json"
  ;;
evaluate)
  POLICY_SHA=$(cd "$SNAPSHOT" && node tools/evaluate-routing-aware-acceptance.js --policy-hash)
  python3 -P "$F/build-evidence.py" "$ROOT" "$ROOT/plan.json" "$POLICY_SHA" "$PROMPTS"
  (cd "$SNAPSHOT" && node tools/evaluate-routing-aware-acceptance.js --evidence "$ROOT/evidence/evidence.json" \
    --out "$ROOT/evidence/verdict-$(date -u +%Y%m%dT%H%M%SZ).json") || true
  ;;
*) echo "usage: run-fresh.sh cpu|gpu|audit|evaluate" >&2; exit 2 ;;
esac
