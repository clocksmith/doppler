#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Manual physical experiment, excluded from ordinary CI test discovery.
: "${DOPPLER_GPU_TIMING_LOCK:?Set the actual shared per-device timing lock path}"
: "${DOPPLER_GPU_TIMING_LOCK_ID:?Set the expected timing lock device:inode}"
[ -d /dev/dri ] || { echo 'No physical /dev/dri; refusing a software run.' >&2; exit 1; }
exec 9<"$DOPPLER_GPU_TIMING_LOCK"
[ "$(stat -Lc '%d:%i' /proc/$$/fd/9)" = "$DOPPLER_GPU_TIMING_LOCK_ID" ]
flock -x 9
[ "$(stat -Lc '%d:%i' "$DOPPLER_GPU_TIMING_LOCK")" = "$DOPPLER_GPU_TIMING_LOCK_ID" ]
export DOPPLER_GROUPED_LOCK_FD=9
node tools/experiments/moe-grouped-physical.js
