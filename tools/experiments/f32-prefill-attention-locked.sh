#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec 9< /run/media/x/models/doe-gpu-timing.lock
[[ "$(stat -Lc '%d:%i' /proc/$$/fd/9)" == '2050:14593' ]] || { echo 'Wrong timing lock inode' >&2; exit 2; }
flock -x 9
export DOPPLER_F32_PREFILL_LOCK_HELD=2050:14593
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
