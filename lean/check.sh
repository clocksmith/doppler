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
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/ExecutionRules.olean" "${ROOT_DIR}/lean/Doppler/ExecutionRules.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/ExecutionRulesFixtures.olean" "${ROOT_DIR}/lean/Doppler/ExecutionRulesFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/ExecutionV0Contract.olean" "${ROOT_DIR}/lean/Doppler/ExecutionV0Contract.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/ExecutionV0ContractFixtures.olean" "${ROOT_DIR}/lean/Doppler/ExecutionV0ContractFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/ExecutionV0Graph.olean" "${ROOT_DIR}/lean/Doppler/ExecutionV0Graph.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/ExecutionV0GraphFixtures.olean" "${ROOT_DIR}/lean/Doppler/ExecutionV0GraphFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/KernelPath.olean" "${ROOT_DIR}/lean/Doppler/KernelPath.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/KernelPathFixtures.olean" "${ROOT_DIR}/lean/Doppler/KernelPathFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/LayerPattern.olean" "${ROOT_DIR}/lean/Doppler/LayerPattern.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/LayerPatternFixtures.olean" "${ROOT_DIR}/lean/Doppler/LayerPatternFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/MergeSemantics.olean" "${ROOT_DIR}/lean/Doppler/MergeSemantics.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/MergeSemanticsFixtures.olean" "${ROOT_DIR}/lean/Doppler/MergeSemanticsFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/Quantization.olean" "${ROOT_DIR}/lean/Doppler/Quantization.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/QuantizationFixtures.olean" "${ROOT_DIR}/lean/Doppler/QuantizationFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/RequiredInferenceFields.olean" "${ROOT_DIR}/lean/Doppler/RequiredInferenceFields.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/RequiredInferenceFieldsFixtures.olean" "${ROOT_DIR}/lean/Doppler/RequiredInferenceFieldsFixtures.lean"
LEAN_PATH="${BUILD_DIR}:${ROOT_DIR}/lean" \
  "${LEAN_BIN}" "+${TOOLCHAIN_REF}" -o "${BUILD_DIR}/Doppler/Check.olean" "${ROOT_DIR}/lean/Doppler/Check.lean"

echo "lean-check: ok (${TOOLCHAIN_REF})"
