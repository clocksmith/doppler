#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN_VERSION="${DOPPLER_LEAN_VERSION:-4.16.0}"

if [[ "${TOOLCHAIN_VERSION}" == v* ]]; then
  TOOLCHAIN_REF="leanprover/lean4:${TOOLCHAIN_VERSION}"
else
  TOOLCHAIN_REF="leanprover/lean4:v${TOOLCHAIN_VERSION}"
fi

if [[ -x "${HOME}/.elan/bin/lean" ]]; then
  LEAN_BIN="${HOME}/.elan/bin/lean"
elif command -v lean >/dev/null 2>&1; then
  LEAN_BIN="$(command -v lean)"
else
  echo "lean binary not found. Install Lean with elan first." >&2
  exit 1
fi

BUILD_DIR="$(mktemp -d /tmp/doppler-lean-check.XXXXXX)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
mkdir -p "${BUILD_DIR}/Doppler"

LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/Model.olean" "${ROOT_DIR}/lean/Doppler/Model.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/ExecutionContract.olean" "${ROOT_DIR}/lean/Doppler/ExecutionContract.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/ExecutionContractFixtures.olean" "${ROOT_DIR}/lean/Doppler/ExecutionContractFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/KernelPath.olean" "${ROOT_DIR}/lean/Doppler/KernelPath.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/KernelPathFixtures.olean" "${ROOT_DIR}/lean/Doppler/KernelPathFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/Check.olean" "${ROOT_DIR}/lean/Doppler/Check.lean"

echo "lean-check: ok (${TOOLCHAIN_REF})"
