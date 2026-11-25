#!/bin/sh
# Autonomous encrypted generation: one session per prompt, each filling the 8-position context, with the key epoch
# reused after the first session (requester, WebGPU evaluator and CPU evaluator all keep it). Run from
# tools/privacy-protected after `go build -o "$PP" .` and `pp gpu-compile` of the session programs.
#
#   PP=/path/to/pp PROGRAMS="/path/a /path/b" OUT=/path/to/dir PROMPTS="h dgj fox" [DUAL=1] [GPU_EVALUATOR=host:port] \
#     [PROFILE=research-n16-u32] [TIMING=per-layer|overlap] [STATE=dir] sh webgpu/run-sessions.sh
#
# STATE (default OUT) holds the requester's, the WebGPU evaluator's and the CPU evaluator's key-epoch stores; point it
# at an earlier run to reuse that key epoch without uploading keys again.
#
# With GPU_EVALUATOR set, the WebGPU evaluator is already running elsewhere (with --sessions covering the prompts).
# With DUAL=1 a CPU evaluator (Lattigo) is the second evaluator of every session; the requester decrypts a step only if
# both results are byte-identical.
set -eu
: "${PP:?}" "${OUT:?}" "${PROMPTS:?}"
PROFILE=${PROFILE:-research-n16-u32}
STATE=${STATE:-$OUT}
CONTEXT=8
mkdir -p "$OUT"
pids=""
trap 'for p in $pids; do kill "$p" 2>/dev/null || true; done' EXIT
count=0
for _ in $PROMPTS; do count=$((count + 1)); done
if [ -z "${GPU_EVALUATOR:-}" ]; then
  : "${PROGRAMS:?}"
  set --
  for p in $PROGRAMS; do set -- "$@" --program "$p"; done
  rm -f "$OUT/ready-gpu"
  node webgpu/evaluator.js "$@" --sessions "$count" --state-dir "$STATE/gpu-state" --ready-file "$OUT/ready-gpu" \
    --report "$OUT/gpu-evaluator.json" --timing "${TIMING:-per-layer}" 2>"$OUT/gpu-evaluator.log" &
  pids="$pids $!"
  until [ -s "$OUT/ready-gpu" ]; do sleep 0.2; done
  GPU_EVALUATOR=$(cat "$OUT/ready-gpu")
fi
INSECURE=""
case "$PROFILE" in *insecure*) INSECURE=--allow-insecure-dev ;; esac
MODEL="sha256:$(shasum -a 256 tiny-model.json | cut -d' ' -f1)"
i=0
status=0
for prompt in $PROMPTS; do
  i=$((i + 1))
  gen=$((CONTEXT - ${#prompt} + 1))
  set -- --evaluator "$GPU_EVALUATOR"
  if [ "${DUAL:-0}" = 1 ]; then
    rm -f "$OUT/ready-cpu"
    "$PP" evaluator --state-dir "$STATE/cpu-state" --ready-file "$OUT/ready-cpu" --report "$OUT/cpu-evaluator-$i.json" \
      2>"$OUT/cpu-evaluator-$i.log" &
    pids="$pids $!"
    until [ -s "$OUT/ready-cpu" ]; do sleep 0.2; done
    set -- "$@" --evaluator "$(cat "$OUT/ready-cpu")"
  fi
  "$PP" requester --profile "$PROFILE" $INSECURE --model-identity "$MODEL" --prompt "$prompt" --generate "$gen" \
    --state-dir "$STATE/requester-state" --report "$OUT/requester-$i.json" "$@" || status=$?
  [ "$status" = 0 ] || break
  "$PP" compare --report "$OUT/requester-$i.json" > "$OUT/compare-$i.json" || true  # nonzero if any reference differs; see the file
done
wait 2>/dev/null || true
exit "$status"
