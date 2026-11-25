#!/bin/sh
# One encrypted token: a requester, a WebGPU evaluator and (optionally) a CPU evaluator as the second side of the
# dual-evaluation check. Run from tools/privacy-protected after `go build -o "$PP" .` and `pp gpu-compile`.
#
#   PP=/path/to/pp PROGRAM=/path/to/prog-base OUT=/path/to/dir [GPU_EVALUATOR=host:port] [DUAL=1] [MISBEHAVE=...] \
#     [PROMPT=hkn] [PROFILE=research-n16-u32] [TIMING=per-layer|overlap] sh webgpu/run-one-token.sh
#
# With GPU_EVALUATOR set, no local WebGPU evaluator is started (it runs on another machine). With DUAL=1 a local CPU
# evaluator (Lattigo) is the second evaluator, and the requester decrypts only if both results are byte-identical.
set -eu
: "${PP:?}" "${OUT:?}"
PROMPT=${PROMPT:-hkn}
PROFILE=${PROFILE:-research-n16-u32}
MISBEHAVE=${MISBEHAVE:-}
mkdir -p "$OUT"
rm -f "$OUT/ready-gpu" "$OUT/ready-cpu"
pids=""
trap 'for p in $pids; do kill "$p" 2>/dev/null || true; done' EXIT
if [ -z "${GPU_EVALUATOR:-}" ]; then
  : "${PROGRAM:?}"
  node webgpu/evaluator.js --program "$PROGRAM" --ready-file "$OUT/ready-gpu" --report "$OUT/gpu-evaluator.json" \
    --misbehave "$MISBEHAVE" --timing "${TIMING:-per-layer}" 2>"$OUT/gpu-evaluator.log" &
  pids="$pids $!"
  until [ -s "$OUT/ready-gpu" ]; do sleep 0.2; done
  GPU_EVALUATOR=$(cat "$OUT/ready-gpu")
fi
set -- --evaluator "$GPU_EVALUATOR"
if [ "${DUAL:-0}" = 1 ]; then
  "$PP" evaluator --ready-file "$OUT/ready-cpu" --report "$OUT/cpu-evaluator.json" 2>"$OUT/cpu-evaluator.log" &
  pids="$pids $!"
  until [ -s "$OUT/ready-cpu" ]; do sleep 0.2; done
  set -- "$@" --evaluator "$(cat "$OUT/ready-cpu")"
fi
INSECURE=""
case "$PROFILE" in *insecure*) INSECURE=--allow-insecure-dev ;; esac
status=0
"$PP" requester --profile "$PROFILE" $INSECURE --model-identity "sha256:$(shasum -a 256 tiny-model.json | cut -d' ' -f1)" \
  --prompt "$PROMPT" --generate 1 --report "$OUT/requester.json" "$@" || status=$?
wait 2>/dev/null || true
if [ "$status" = 0 ]; then "$PP" compare --report "$OUT/requester.json" > "$OUT/compare.json"; fi
exit "$status"
