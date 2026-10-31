#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
: "${DOPPLER_GPU_TIMING_LOCK:?set DOPPLER_GPU_TIMING_LOCK to the host GPU timing lock file}"
exec 9< "$DOPPLER_GPU_TIMING_LOCK"
lock_id="$(stat -Lc '%d:%i' /proc/$$/fd/9)"
if [[ -n "${DOPPLER_GPU_TIMING_LOCK_ID:-}" && "$lock_id" != "$DOPPLER_GPU_TIMING_LOCK_ID" ]]; then
  echo 'Wrong timing lock inode' >&2; exit 2
fi
flock -x 9
export DOPPLER_F32_PREFILL_LOCK_HELD="$lock_id"
case "${1:-}" in
  static)
    node --check tools/experiments/f32-prefill-attention-physical.js
    node --check tools/experiments/prepare-f32-prefill-attention.js
    node tests/kernels/attention-f32-prefill-selection.test.js
    node tools/generate-kernel-uniforms.js --check
    node tools/sync-kernel-ref-digests.js --check
    ;;
  prepare)
    node tools/experiments/prepare-f32-prefill-attention.js \
      PREFILL_PROFILE_RUN/model-02/input-config.json F32_PREFILL_ATTENTION_REVIEW/candidate
    ;;
  physical)
    node tools/experiments/f32-prefill-attention-physical.js --adapter-match "${2:-radeon}" --bench
    ;;
  *) echo 'Usage: bash tools/experiments/f32-prefill-attention-locked.sh static|prepare|physical [adapter-match]' >&2; exit 2 ;;
esac
