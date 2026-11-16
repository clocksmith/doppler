import { getLayout, getWeightDtype } from "../../../../gpu/weight-buffer.js";
import { castF16ToF32, castF32ToF16, recordCastF16ToF32, recordCastF32ToF16 } from "../../../../gpu/kernels/cast.js";
import { selectRuleValue } from "../../../../rules/rule-registry.js";
import { QK_K, Q4K_BLOCK_BYTES } from "../../../../config/schema/index.js";
import { getKernelPathMatmulPrecision } from "../../../../config/kernel-path-loader.js";

function normalizeProjectionMatmulDtype(value, precisionField = 'dtype') {
  if (value == null || value === '') {
    return null;
  }
  const normalized = String(value).trim().toLowerCase();
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(
      `[ExecutionV1] attention projection ${precisionField} must be "f16" or "f32"; got "${value}".`
    );
  }
  return normalized;
}

export async function coerceProjectionInputTensor(recorder, tensor, targetDtype) {
  if (!targetDtype || tensor.dtype === targetDtype) {
    return tensor;
  }
  if (targetDtype === 'f16') {
    return recorder
      ? recordCastF32ToF16(recorder, tensor)
      : castF32ToF16(tensor);
  }
  return recorder
    ? recordCastF16ToF32(recorder, tensor)
    : castF16ToF32(tensor);
}

export function resolveProjectionMatmulDtype({
  useFusedQKV,
  phase,
  layerIdx,
  kernelPath,
  precisionField,
  fallbackDtype,
}) {
  const roles = useFusedQKV ? ['qkv_proj'] : ['q_proj', 'k_proj', 'v_proj'];
  const explicitInputDtypes = roles
    .map((role) => normalizeProjectionMatmulDtype(
      getKernelPathMatmulPrecision(role, phase, layerIdx, kernelPath)?.[precisionField] ?? null,
      precisionField
    ))
    .filter(Boolean);
  if (explicitInputDtypes.length === 0) {
    return fallbackDtype;
  }
  const [resolvedInputDtype] = explicitInputDtypes;
  if (explicitInputDtypes.some((dtype) => dtype !== resolvedInputDtype)) {
    throw new Error(
      `[ExecutionV1] attention projection steps resolved conflicting ${precisionField} values at layer ${layerIdx}: ` +
      `${explicitInputDtypes.join(', ')}.`
    );
  }
  return resolvedInputDtype;
}

export function resolveProjectionInputDtype({ useFusedQKV, phase, layerIdx, kernelPath, fallbackDtype }) {
  return resolveProjectionMatmulDtype({
    useFusedQKV,
    phase,
    layerIdx,
    kernelPath,
    precisionField: 'inputDtype',
    fallbackDtype,
  });
}

export function resolveProjectionOutputDtype({ useFusedQKV, phase, layerIdx, kernelPath, fallbackDtype }) {
  return resolveProjectionMatmulDtype({
    useFusedQKV,
    phase,
    layerIdx,
    kernelPath,
    precisionField: 'outputDtype',
    fallbackDtype,
  });
}

export function resolveProjectionSliceOffsetBytes(weightBuffer, outputRows, inputCols) {
  const safeRows = Number.isFinite(outputRows) ? Math.max(0, Math.floor(outputRows)) : 0;
  const safeCols = Number.isFinite(inputCols) ? Math.max(0, Math.floor(inputCols)) : 0;
  if (safeRows === 0 || safeCols === 0) {
    return 0;
  }

  const dtype = String(getWeightDtype(weightBuffer) ?? '').toLowerCase();
  if (dtype === 'q4k') {
    const layout = String(getLayout(weightBuffer) ?? 'row').toLowerCase();
    if (layout !== 'row') {
      throw new Error(`resolveProjectionSliceOffsetBytes: unsupported q4k layout "${layout}" for projection slicing.`);
    }
    const blocksPerRow = Math.ceil(safeCols / QK_K);
    const bytesPerRow = blocksPerRow * Q4K_BLOCK_BYTES;
    return safeRows * bytesPerRow;
  }

  if (dtype === 'f16' || dtype === 'bf16') {
    return safeRows * safeCols * 2;
  }
  return safeRows * safeCols * 4;
}

export function shouldForceF32AttentionProjectionForRoPE({
  attentionInputDtype,
  headDim,
  rotaryDim = headDim,
  interleaved = false,
  kernelPathIsF16 = false,
}) {
  // When the execution graph specifies f16 matmul kernels for Q/K/V projections,
  // the graph is authoritative. The f16 RoPE kernel handles partial rotation and
  // interleaving at f16 precision. Do not override to f32.
  if (kernelPathIsF16) return false;
  return attentionInputDtype === 'f16'
    && Number.isFinite(headDim)
    && Number.isFinite(rotaryDim)
    && (rotaryDim !== headDim || interleaved === true);
}

export function resolveAttentionProjectionOutputDtype(attentionInputDtype, options = {}) {
  const useF16Activations = attentionInputDtype === 'f16';
  return selectRuleValue('inference', 'dtype', 'attentionProjectionOutputDtype', {
    forceF32: options.forceF32 === true,
    useF16: useF16Activations,
    fallback: attentionInputDtype,
  });
}
