import { getBufferDtype, isGpuBufferInstance, isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import {
  recordMatmul,
  recordRMSNorm,
  runMatmul,
  runRMSNorm,
  castF32ToF16,
  castF16ToF32,
  recordCastF32ToF16,
  recordCastF16ToF32,
} from '../../../../gpu/kernel-selector.js';
import { readBuffer, releaseBuffer, uploadData, acquireBuffer } from '../../../../memory/buffer-pool.js';
import { log } from '../../../../debug/index.js';
import { decodeReadback, f16ToF32 } from '../debug-utils/index.js';
import { runLinearAttentionCoreGPU } from '../../../../gpu/kernels/linear-attention-core.js';
import {
  resolveLinearAttentionABProjection,
  resolveLinearAttentionQKVZProjection,
} from '../linear-attention-ab-fusion.js';
import { runProbes } from '../probes.js';
import { QK_K, Q4K_BLOCK_BYTES } from '../../../../config/schema/index.js';
import { dequantizeQ4KM } from '../../../../converter/quantizer.js';
import { getKernelPathMatmulPrecision } from '../../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';

export function isGpuBuffer(value) {
  return isGpuBufferInstance(value);
}

export function toPositiveInt(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return null;
  return Math.trunc(num);
}

export function normalizeLinearNormMode(value) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (normalized === 'shared') return 'shared';
  if (normalized === 'per_head' || normalized === 'per-head' || normalized === 'perhead') {
    return 'per_head';
  }
  return null;
}

export function bytesFromDtype(dtype) {
  return selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
}

export function resolveMatmulStepDtype(role, phase, layerIdx, kernelPath, fallback, field) {
  const precision = getKernelPathMatmulPrecision(role, phase, layerIdx, kernelPath);
  const requested = precision?.[field] ?? fallback;
  if (requested == null) {
    return fallback;
  }
  return selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: requested });
}

export function applyLinearNormWeightOffset(values, rmsNormWeightOffset) {
  if (!(values instanceof Float32Array)) {
    throw new Error('applyLinearNormWeightOffset requires Float32Array input.');
  }
  // Qwen linear-attention output norm uses direct weights (values ~1.0,
  // standard RMSNorm initialization) even when surrounding transformer
  // RMSNorm sites use the Gemma-style (1 + weight) formula (values ~0.24).
  // Verified from Qwen 3.5 checkpoint: linear_attn.norm.weight mean≈0.95.
  return values;
}

export function isResolvedWeightShared(originalWeight) {
  return isGpuBuffer(originalWeight) || isWeightBuffer(originalWeight);
}

export function releaseOrTrackBuffer(recorder, buffer) {
  if (!isGpuBuffer(buffer)) return;
  if (recorder && typeof recorder.trackTemporaryBuffer === 'function') {
    recorder.trackTemporaryBuffer(buffer);
  } else {
    releaseBuffer(buffer);
  }
}

export async function applyProjectionLoRA({
  inputTensor,
  baseOutput,
  loraModule,
  dims,
  getWeightBuffer,
  recorder,
  kernelPath,
}) {
  if (!loraModule) {
    return baseOutput;
  }
  try {
    const combined = await applyLoRA(
      inputTensor,
      baseOutput,
      loraModule,
      dims,
      getWeightBuffer,
      recorder ?? undefined,
      { kernelPath }
    );
    if (combined.buffer !== baseOutput.buffer) {
      releaseOrTrackBuffer(recorder, baseOutput.buffer);
    }
    return combined;
  } catch (error) {
    releaseOrTrackBuffer(recorder, baseOutput.buffer);
    throw error;
  }
}

export function releaseResolvedWeightBuffer(originalWeight, resolvedWeight, recorder) {
  if (isResolvedWeightShared(originalWeight)) {
    return;
  }
  const resolvedBuffer = isWeightBuffer(resolvedWeight) ? resolvedWeight.buffer : resolvedWeight;
  releaseOrTrackBuffer(recorder, resolvedBuffer);
}

export function inferLinearNormModeFromWeight(weight, projectionLayout) {
  const sharedElements = projectionLayout.headVDim;
  const perHeadElements = projectionLayout.valueDim;
  const classify = (length) => {
    if (!Number.isFinite(length) || length <= 0) return null;
    const elements = Math.trunc(length);
    if (elements === sharedElements) return 'shared';
    if (elements === perHeadElements) return 'per_head';
    return null;
  };

  if (isWeightBuffer(weight) && Array.isArray(weight.shape) && weight.shape.length > 0) {
    const elements = weight.shape.reduce(
      (total, dim) => total * Math.max(1, Math.trunc(Number(dim) || 0)),
      1
    );
    return classify(elements);
  }
  if (weight instanceof Float32Array || weight instanceof Float64Array) {
    return classify(weight.length);
  }
  if (weight instanceof Uint16Array || weight instanceof Int16Array) {
    return classify(weight.length);
  }
  if (ArrayBuffer.isView(weight)) {
    return classify(weight.length);
  }
  if (weight instanceof ArrayBuffer) {
    return classify(Math.trunc(weight.byteLength / Float32Array.BYTES_PER_ELEMENT));
  }
  if (!weight || typeof weight !== 'object') {
    return null;
  }
  const explicitDtype = typeof weight?.dtype === 'string' ? weight.dtype.toLowerCase() : null;
  const trackedDtype = isGpuBuffer(weight) ? String(getBufferDtype(weight) ?? '').toLowerCase() : '';
  const sourceDtype = explicitDtype || trackedDtype;
  if (!sourceDtype || !Number.isFinite(weight?.size)) {
    return null;
  }
  const bytesPerElement = bytesFromDtype(sourceDtype);
  const size = Number(weight.size);
  const sizedElements = Math.trunc(size / bytesPerElement);
  if (sizedElements && size % bytesPerElement === 0) {
    return classify(sizedElements);
  }
  return null;
}

export function resolveLinearNormMode(configNormMode, normWeight, projectionLayout, layerIdx) {
  const configuredMode = normalizeLinearNormMode(configNormMode);
  const inferredMode = inferLinearNormModeFromWeight(normWeight, projectionLayout);
  if (configuredMode && inferredMode && configuredMode !== inferredMode) {
    throw new Error(
      `linear_attention layer ${layerIdx} declares linearNormMode="${configuredMode}" ` +
      `but norm.weight shape implies "${inferredMode}".`
    );
  }
  if (configuredMode) {
    return configuredMode;
  }
  if (inferredMode) {
    return inferredMode;
  }
  throw new Error(
    `linear_attention layer ${layerIdx} requires explicit linearNormMode or a norm.weight shape that resolves it.`
  );
}

export async function readWeightAsF32(weight, expectedElements, label) {
  if (weight == null) {
    throw new Error(`Missing linear_attention weight: ${label}`);
  }

  if (weight instanceof Float32Array) {
    if (expectedElements != null && weight.length !== expectedElements) {
      throw new Error(
        `Weight "${label}" has ${weight.length} elements, expected ${expectedElements}.`
      );
    }
    return weight.slice();
  }

  if (ArrayBuffer.isView(weight)) {
    let copied;
    if (weight instanceof Uint16Array || weight instanceof Int16Array) {
      const raw = new Uint16Array(weight.buffer, weight.byteOffset, weight.byteLength / 2);
      copied = new Float32Array(raw.length);
      for (let index = 0; index < raw.length; index += 1) {
        copied[index] = f16ToF32(raw[index]);
      }
    } else if (
      weight instanceof Float64Array
      || weight instanceof Float32Array
      || weight instanceof Int32Array
      || weight instanceof Uint32Array
    ) {
      copied = Float32Array.from(weight);
    } else {
      throw new Error(
        `Unsupported typed-array view for "${label}": ${weight.constructor?.name ?? 'Unknown'}.`
      );
    }
    if (expectedElements != null && copied.length !== expectedElements) {
      throw new Error(
        `Weight "${label}" has ${copied.length} elements, expected ${expectedElements}.`
      );
    }
    return copied;
  }

  if (weight instanceof ArrayBuffer) {
    let copied;
    if (expectedElements != null && weight.byteLength === expectedElements * 2) {
      copied = decodeReadback(weight, 'f16');
    } else {
      copied = new Float32Array(weight.slice(0));
    }
    if (expectedElements != null && copied.length !== expectedElements) {
      throw new Error(
        `Weight "${label}" has ${copied.length} elements, expected ${expectedElements}.`
      );
    }
    return copied;
  }

  let sourceBuffer = null;
  let sourceDtype = null;
  if (isWeightBuffer(weight)) {
    sourceBuffer = weight.buffer;
    sourceDtype = String(weight.dtype ?? '').toLowerCase();
  } else if (isGpuBuffer(weight)) {
    sourceBuffer = weight;
    sourceDtype = String(getBufferDtype(weight) ?? '').toLowerCase();
  }

  if (!sourceBuffer) {
    throw new Error(`Unsupported weight type for "${label}".`);
  }

  let elementCount = expectedElements;
  if (!elementCount && isWeightBuffer(weight) && Array.isArray(weight.shape) && weight.shape.length > 0) {
    elementCount = weight.shape.reduce((total, dim) => total * Math.max(1, Math.trunc(Number(dim) || 0)), 1);
  }
  const isQ4K = sourceDtype === 'q4k' || sourceDtype === 'q4_k_m' || sourceDtype === 'q4_k';
  if (!elementCount) {
    if (isQ4K) {
      elementCount = Math.trunc(sourceBuffer.size / Q4K_BLOCK_BYTES) * QK_K;
    } else {
      const inferredBytes = sourceDtype === 'f16' || sourceDtype === 'bf16' ? 2 : 4;
      elementCount = Math.trunc(sourceBuffer.size / inferredBytes);
    }
  }

  if (isQ4K) {
    const numBlocks = Math.ceil(elementCount / QK_K);
    const q4kBytes = numBlocks * Q4K_BLOCK_BYTES;
    const raw = await readBuffer(sourceBuffer, q4kBytes);
    const decoded = dequantizeQ4KM(new Uint8Array(raw), numBlocks, [elementCount]);
    if (expectedElements != null && decoded.length !== expectedElements) {
      throw new Error(
        `Weight "${label}" Q4K decoded length ${decoded.length}, expected ${expectedElements}.`
      );
    }
    return decoded;
  }

  if (!sourceDtype) {
    const bytesPer = sourceBuffer.size / elementCount;
    sourceDtype = bytesPer <= 2 ? 'f16' : 'f32';
  }

  const readBytes = elementCount * bytesFromDtype(sourceDtype);
  const raw = await readBuffer(sourceBuffer, readBytes);
  const decoded = decodeReadback(raw, sourceDtype);
  if (expectedElements != null && decoded.length !== expectedElements) {
    throw new Error(
      `Weight "${label}" decoded length ${decoded.length}, expected ${expectedElements}.`
    );
  }
  return decoded;
}

export function uploadF32Buffer(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}

export function ensureLayerRuntimeGpuBuffers(layerState) {
  if (!isGpuBuffer(layerState.convWeightGPU)) {
    layerState.convWeightGPU = uploadF32Buffer(layerState.convWeight, `L${layerState.layerIdx}.linear_conv_weight`);
  }
  if (!isGpuBuffer(layerState.dtBiasGPU)) {
    layerState.dtBiasGPU = uploadF32Buffer(layerState.dtBias, `L${layerState.layerIdx}.linear_dt_bias`);
  }
  if (!isGpuBuffer(layerState.aNegExpGPU)) {
    layerState.aNegExpGPU = uploadF32Buffer(layerState.aNegExp, `L${layerState.layerIdx}.linear_a_neg_exp`);
  }
  if (!isGpuBuffer(layerState.normWeightGPU)) {
    layerState.normWeightGPU = uploadF32Buffer(layerState.normWeight, `L${layerState.layerIdx}.linear_norm_weight`);
  }
  if (!isGpuBuffer(layerState.convStateGPU)) {
    layerState.convStateGPU = uploadF32Buffer(layerState.convState, `L${layerState.layerIdx}.linear_conv_state`);
  }
  if (!isGpuBuffer(layerState.recurrentStateGPU)) {
    layerState.recurrentStateGPU = uploadF32Buffer(layerState.recurrentState, `L${layerState.layerIdx}.linear_recurrent_state`);
  }
}

export async function createLayerRuntimeState(
  layerIdx,
  layerWeights,
  config,
  currentSeqLen,
  projectionLayout
) {
  const convKernel = layerWeights.linearConv1D;
  const dtBiasWeight = layerWeights.linearDtBias;
  const aLogWeight = layerWeights.linearALog;
  const normWeight = layerWeights.linearNorm;

  if (!convKernel || !dtBiasWeight || !aLogWeight || !normWeight) {
    throw new Error(
      `linear_attention layer ${layerIdx} is missing one or more required weights: ` +
      'conv1d, dt_bias, A_log, norm.'
    );
  }

  let convKernelSize = toPositiveInt(config.linearConvKernelDim) ?? null;
  if (isWeightBuffer(convKernel) && Array.isArray(convKernel.shape) && convKernel.shape.length >= 3) {
    const shapeKernelSize = toPositiveInt(convKernel.shape[2]) ?? null;
    if (convKernelSize != null && shapeKernelSize != null && convKernelSize !== shapeKernelSize) {
      throw new Error(
        `linear_attention layer ${layerIdx} declares linearConvKernelDim=${convKernelSize}, ` +
        `but conv1d weight shape implies ${shapeKernelSize}.`
      );
    }
    convKernelSize = shapeKernelSize ?? convKernelSize;
  }
  if (!convKernelSize) {
    throw new Error(`linear_attention layer ${layerIdx} requires linearConvKernelDim.`);
  }

  const convWeight = await readWeightAsF32(
    convKernel,
    projectionLayout.convDim * convKernelSize,
    `L${layerIdx}.linear_attn.conv1d.weight`
  );
  const dtBias = await readWeightAsF32(
    dtBiasWeight,
    projectionLayout.numVHeads,
    `L${layerIdx}.linear_attn.dt_bias`
  );
  const aLog = await readWeightAsF32(
    aLogWeight,
    projectionLayout.numVHeads,
    `L${layerIdx}.linear_attn.A_log`
  );
  const normMode = resolveLinearNormMode(config.linearNormMode, normWeight, projectionLayout, layerIdx);
  const expectedNormElements = normMode === 'per_head'
    ? projectionLayout.valueDim
    : projectionLayout.headVDim;
  const norm = await readWeightAsF32(
    normWeight,
    expectedNormElements,
    `L${layerIdx}.linear_attn.norm.weight`
  );
  const runtimeNorm = applyLinearNormWeightOffset(norm, config.rmsNormWeightOffset === true);

  const aNegExp = new Float32Array(aLog.length);
  for (let i = 0; i < aLog.length; i++) {
    aNegExp[i] = -Math.exp(aLog[i]);
  }

  const convState = new Float32Array(projectionLayout.convDim * convKernelSize);
  const recurrentState = new Float32Array(
    projectionLayout.numVHeads * projectionLayout.headKDim * projectionLayout.headVDim
  );
  const rmsNormEps = Number(config.rmsNormEps);
  if (!Number.isFinite(rmsNormEps) || rmsNormEps <= 0) {
    throw new Error(`linear_attention layer ${layerIdx} requires a positive rmsNormEps.`);
  }

  const layerState = {
    layerIdx,
    seqLen: currentSeqLen,
    warnedSeqMismatch: false,
    convKernelSize,
    convDim: projectionLayout.convDim,
    keyDim: projectionLayout.keyDim,
    valueDim: projectionLayout.valueDim,
    numKHeads: projectionLayout.numKHeads,
    numVHeads: projectionLayout.numVHeads,
    headKDim: projectionLayout.headKDim,
    headVDim: projectionLayout.headVDim,
    qSize: projectionLayout.qSize,
    kSize: projectionLayout.kSize,
    vSize: projectionLayout.vSize,
    qRep: projectionLayout.qRep,
    normMode,
    rmsNormEps,
    convWeight,
    dtBias,
    aNegExp,
    normWeight: runtimeNorm,
    convState,
    recurrentState,
    convWeightGPU: null,
    dtBiasGPU: null,
    aNegExpGPU: null,
    normWeightGPU: null,
    convStateGPU: null,
    recurrentStateGPU: null,
  };

  ensureLayerRuntimeGpuBuffers(layerState);
  return layerState;
}

export async function projectLinearTensor({
  inputTensor,
  sourceWeight,
  role,
  phase,
  outDim,
  numTokens,
  hiddenSize,
  layerIdx,
  kernelPath,
  outputDtype,
  getWeightBuffer,
  recorder,
  executionPolicies = null,
  loraModule = null,
}) {
  const resolvedWeight = getWeightBuffer(sourceWeight, role);
  const resolvedInputDtype = resolveMatmulStepDtype(
    role,
    phase,
    layerIdx,
    kernelPath,
    inputTensor.dtype,
    'inputDtype'
  );
  const resolvedOutputDtype = resolveMatmulStepDtype(
    role,
    phase,
    layerIdx,
    kernelPath,
    outputDtype,
    'outputDtype'
  );
  const wantsF16Input = inputTensor.dtype === 'f32' && resolvedInputDtype === 'f16';
  let matmulInput = inputTensor;
  if (wantsF16Input) {
    assertImplicitDtypeTransitionAllowed({
      executionPolicies,
      fromDtype: inputTensor.dtype,
      toDtype: 'f16',
      op: role,
      detail: 'Linear attention projection would narrow activations implicitly.',
    });
    matmulInput = recorder
      ? await recordCastF32ToF16(recorder, inputTensor)
      : await castF32ToF16(inputTensor);
  }
  const settleProjectionOutputDtype = async (result) => {
    if (result.dtype === resolvedOutputDtype) {
      return result;
    }
    assertImplicitDtypeTransitionAllowed({
      executionPolicies,
      fromDtype: result.dtype,
      toDtype: resolvedOutputDtype,
      op: role,
      detail: 'Linear attention projection returned a kernel-selected dtype different from the declared projection output dtype.',
    });
    if (resolvedOutputDtype === 'f16') {
      const casted = recorder ? await recordCastF32ToF16(recorder, result) : await castF32ToF16(result);
      releaseOrTrackBuffer(recorder, result.buffer);
      return casted;
    }
    if (resolvedOutputDtype === 'f32') {
      const casted = recorder ? await recordCastF16ToF32(recorder, result) : await castF16ToF32(result);
      releaseOrTrackBuffer(recorder, result.buffer);
      return casted;
    }
    throw new Error(`Unsupported linear_attention projection output dtype "${String(resolvedOutputDtype)}".`);
  };
  try {
    if (recorder) {
      const result = await recordMatmul(recorder, matmulInput, resolvedWeight, numTokens, outDim, hiddenSize, {
        transposeB: 'auto',
        role,
        layerIdx,
        kernelPath,
        outputDtype: resolvedOutputDtype,
        executionPolicies,
      });
      const projected = await settleProjectionOutputDtype(result);
      return await applyProjectionLoRA({
        inputTensor: matmulInput,
        baseOutput: projected,
        loraModule,
        dims: { M: numTokens, N: outDim, K: hiddenSize },
        getWeightBuffer,
        recorder,
        kernelPath,
      });
    }
    const result = await runMatmul(matmulInput, resolvedWeight, numTokens, outDim, hiddenSize, {
      transposeB: 'auto',
      role,
      layerIdx,
      kernelPath,
      outputDtype: resolvedOutputDtype,
      executionPolicies,
    });
    const projected = await settleProjectionOutputDtype(result);
    return await applyProjectionLoRA({
      inputTensor: matmulInput,
      baseOutput: projected,
      loraModule,
      dims: { M: numTokens, N: outDim, K: hiddenSize },
      getWeightBuffer,
      recorder: null,
      kernelPath,
    });
  } finally {
    if (matmulInput !== inputTensor) {
      releaseOrTrackBuffer(recorder, matmulInput.buffer);
    }
    releaseResolvedWeightBuffer(sourceWeight, resolvedWeight, recorder);
  }
}
