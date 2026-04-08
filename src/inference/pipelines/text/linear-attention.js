import { getBufferDtype, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { recordMatmul, recordRMSNorm, runMatmul, runRMSNorm, castF32ToF16, recordCastF32ToF16 } from '../../../gpu/kernel-selector.js';
import { readBuffer, releaseBuffer, uploadData, acquireBuffer } from '../../../memory/buffer-pool.js';
import { log } from '../../../debug/index.js';
import { decodeReadback } from './debug-utils/index.js';
import { runLinearAttentionCoreGPU } from '../../../gpu/kernels/linear-attention-core.js';
import { runProbes } from './probes.js';
import { QK_K, Q4K_BLOCK_BYTES } from '../../../config/schema/index.js';
import { dequantizeQ4KM } from '../../../converter/quantizer.js';
import { getKernelPathMatmulPrecision } from '../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { assertImplicitDtypeTransitionAllowed } from './dtype-contract.js';

const LINEAR_RUNTIME_SCHEMA_VERSION = 1;
const QK_L2NORM_EPS = 1e-6;

function isGpuBuffer(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}

function toPositiveInt(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return null;
  return Math.trunc(num);
}

function normalizeLinearNormMode(value) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (normalized === 'shared') return 'shared';
  if (normalized === 'per_head' || normalized === 'per-head' || normalized === 'perhead') {
    return 'per_head';
  }
  return null;
}

function bytesFromDtype(dtype) {
  const normalized = String(dtype ?? '').toLowerCase();
  if (normalized === 'f16' || normalized === 'bf16') return 2;
  return 4;
}

function resolveMatmulStepDtype(role, phase, layerIdx, kernelPath, fallback, field) {
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

function cloneLayerRuntimeState(layerState) {
  return {
    layerIdx: layerState.layerIdx,
    seqLen: layerState.seqLen,
    warnedSeqMismatch: layerState.warnedSeqMismatch === true,
    convKernelSize: layerState.convKernelSize,
    convDim: layerState.convDim,
    keyDim: layerState.keyDim,
    valueDim: layerState.valueDim,
    numKHeads: layerState.numKHeads,
    numVHeads: layerState.numVHeads,
    headKDim: layerState.headKDim,
    headVDim: layerState.headVDim,
    qSize: layerState.qSize,
    kSize: layerState.kSize,
    vSize: layerState.vSize,
    qRep: layerState.qRep,
    normMode: layerState.normMode === 'per_head' ? 'per_head' : 'shared',
    rmsNormEps: layerState.rmsNormEps,
    convWeight: layerState.convWeight.slice(),
    dtBias: layerState.dtBias.slice(),
    aNegExp: layerState.aNegExp.slice(),
    normWeight: layerState.normWeight.slice(),
    convState: layerState.convState.slice(),
    recurrentState: layerState.recurrentState.slice(),
  };
}

function cloneLayerMap(layers) {
  const cloned = new Map();
  for (const [layerIdx, layerState] of layers.entries()) {
    cloned.set(layerIdx, cloneLayerRuntimeState(layerState));
  }
  return cloned;
}

function ensureRuntime(runtime) {
  if (runtime && typeof runtime === 'object' && runtime.layers instanceof Map) {
    runtime.schemaVersion = LINEAR_RUNTIME_SCHEMA_VERSION;
    return runtime;
  }
  return createLinearAttentionRuntime();
}

function resolveProjectionLayout(config, layerWeights) {
  const numKHeads = toPositiveInt(config.linearNumKeyHeads);
  const numVHeads = toPositiveInt(config.linearNumValueHeads);
  const headKDim = toPositiveInt(config.linearKeyHeadDim);
  const headVDim = toPositiveInt(config.linearValueHeadDim);
  if (!numKHeads || !numVHeads || !headKDim || !headVDim) {
    throw new Error(
      'linear_attention requires linear_num_key_heads, linear_num_value_heads, ' +
      'linear_key_head_dim, and linear_value_head_dim.'
    );
  }
  if (numVHeads % numKHeads !== 0) {
    throw new Error(
      `linear_attention requires num_value_heads divisible by num_key_heads; got ` +
      `${numVHeads} and ${numKHeads}.`
    );
  }

  const keyDim = numKHeads * headKDim;
  const valueDim = numVHeads * headVDim;
  const qSize = toPositiveInt(layerWeights?.qkvSizes?.[0]) ?? keyDim;
  const kSize = toPositiveInt(layerWeights?.qkvSizes?.[1]) ?? keyDim;
  const vSize = toPositiveInt(layerWeights?.qkvSizes?.[2]) ?? valueDim;
  if (qSize !== keyDim || kSize !== keyDim || vSize !== valueDim) {
    throw new Error(
      `linear_attention projection mismatch: expected [${keyDim}, ${keyDim}, ${valueDim}] ` +
      `but got [${qSize}, ${kSize}, ${vSize}].`
    );
  }

  return {
    numKHeads,
    numVHeads,
    headKDim,
    headVDim,
    keyDim,
    valueDim,
    qSize,
    kSize,
    vSize,
    qRep: numVHeads / numKHeads,
    convDim: qSize + kSize + vSize,
  };
}

function isResolvedWeightShared(originalWeight) {
  return isGpuBuffer(originalWeight) || isWeightBuffer(originalWeight);
}

function releaseOrTrackBuffer(recorder, buffer) {
  if (!isGpuBuffer(buffer)) return;
  if (recorder && typeof recorder.trackTemporaryBuffer === 'function') {
    recorder.trackTemporaryBuffer(buffer);
  } else {
    releaseBuffer(buffer);
  }
}

function releaseResolvedWeightBuffer(originalWeight, resolvedWeight, recorder) {
  if (isResolvedWeightShared(originalWeight)) {
    return;
  }
  const resolvedBuffer = isWeightBuffer(resolvedWeight) ? resolvedWeight.buffer : resolvedWeight;
  releaseOrTrackBuffer(recorder, resolvedBuffer);
}

function inferLinearNormModeFromWeight(weight, projectionLayout) {
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
  const explicitDtype = typeof weight?.dtype === 'string' ? weight.dtype.toLowerCase() : null;
  const trackedDtype = isGpuBuffer(weight) ? String(getBufferDtype(weight) ?? '').toLowerCase() : '';
  const bytesPerElement = bytesFromDtype(explicitDtype || trackedDtype || null);
  const sizedElements = Number.isFinite(weight?.size)
    ? Math.trunc(Number(weight.size) / bytesPerElement)
    : null;
  if (sizedElements && Number(weight.size) % bytesPerElement === 0) {
    return classify(sizedElements);
  }
  return null;
}

export function inferLinearNormMode(weight, projectionLayout) {
  return inferLinearNormModeFromWeight(weight, projectionLayout);
}

function resolveLinearNormMode(configNormMode, normWeight, projectionLayout, layerIdx) {
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

async function readWeightAsF32(weight, expectedElements, label) {
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
      const bytes = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
      copied = decodeReadback(bytes, 'f16');
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
      copied = decodeReadback(weight.slice(0), 'f16');
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

function clearDynamicLayerState(layerState) {
  layerState.convState.fill(0);
  layerState.recurrentState.fill(0);
  if (isGpuBuffer(layerState.convStateGPU)) {
    uploadData(layerState.convStateGPU, layerState.convState);
  }
  if (isGpuBuffer(layerState.recurrentStateGPU)) {
    uploadData(layerState.recurrentStateGPU, layerState.recurrentState);
  }
}

function uploadF32Buffer(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}

function ensureLayerRuntimeGpuBuffers(layerState) {
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

async function syncLayerRuntimeStateFromGPU(layerState) {
  if (isGpuBuffer(layerState.convStateGPU)) {
    const rawConvState = await readBuffer(
      layerState.convStateGPU,
      layerState.convState.length * Float32Array.BYTES_PER_ELEMENT
    );
    layerState.convState = decodeReadback(rawConvState, 'f32');
  }
  if (isGpuBuffer(layerState.recurrentStateGPU)) {
    const rawRecurrentState = await readBuffer(
      layerState.recurrentStateGPU,
      layerState.recurrentState.length * Float32Array.BYTES_PER_ELEMENT
    );
    layerState.recurrentState = decodeReadback(rawRecurrentState, 'f32');
  }
}

function releaseLayerRuntimeGpuBuffers(layerState) {
  if (!layerState || typeof layerState !== 'object') return;
  if (isGpuBuffer(layerState.convWeightGPU)) {
    releaseBuffer(layerState.convWeightGPU);
    layerState.convWeightGPU = null;
  }
  if (isGpuBuffer(layerState.dtBiasGPU)) {
    releaseBuffer(layerState.dtBiasGPU);
    layerState.dtBiasGPU = null;
  }
  if (isGpuBuffer(layerState.aNegExpGPU)) {
    releaseBuffer(layerState.aNegExpGPU);
    layerState.aNegExpGPU = null;
  }
  if (isGpuBuffer(layerState.normWeightGPU)) {
    releaseBuffer(layerState.normWeightGPU);
    layerState.normWeightGPU = null;
  }
  if (isGpuBuffer(layerState.convStateGPU)) {
    releaseBuffer(layerState.convStateGPU);
    layerState.convStateGPU = null;
  }
  if (isGpuBuffer(layerState.recurrentStateGPU)) {
    releaseBuffer(layerState.recurrentStateGPU);
    layerState.recurrentStateGPU = null;
  }
}

function releaseRuntimeLayerBuffers(runtime) {
  if (!runtime || typeof runtime !== 'object' || !(runtime.layers instanceof Map)) {
    return;
  }
  for (const layerState of runtime.layers.values()) {
    releaseLayerRuntimeGpuBuffers(layerState);
  }
}

async function createLayerRuntimeState(
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

function isLayerRuntimeCompatible(layerState, projectionLayout, requestedNormMode = null) {
  return layerState
    && layerState.convDim === projectionLayout.convDim
    && Number.isFinite(layerState.convKernelSize)
    && layerState.convKernelSize > 0
    && layerState.keyDim === projectionLayout.keyDim
    && layerState.valueDim === projectionLayout.valueDim
    && layerState.numKHeads === projectionLayout.numKHeads
    && layerState.numVHeads === projectionLayout.numVHeads
    && layerState.headKDim === projectionLayout.headKDim
    && layerState.headVDim === projectionLayout.headVDim
    && layerState.qRep === projectionLayout.qRep
    && layerState.qSize === projectionLayout.qSize
    && layerState.kSize === projectionLayout.kSize
    && layerState.vSize === projectionLayout.vSize
    && (layerState.normMode === 'shared' || layerState.normMode === 'per_head')
    && (requestedNormMode == null || layerState.normMode === requestedNormMode);
}

async function getLayerRuntimeState(runtime, layerIdx, layerWeights, config, currentSeqLen, projectionLayout) {
  const requestedNormMode = normalizeLinearNormMode(config.linearNormMode);
  let layerState = runtime.layers.get(layerIdx) ?? null;
  if (!isLayerRuntimeCompatible(layerState, projectionLayout, requestedNormMode)) {
    if (layerState) {
      releaseLayerRuntimeGpuBuffers(layerState);
    }
    layerState = await createLayerRuntimeState(
      layerIdx,
      layerWeights,
      config,
      currentSeqLen,
      projectionLayout
    );
    runtime.layers.set(layerIdx, layerState);
    ensureLayerRuntimeGpuBuffers(layerState);
    return layerState;
  }

  if (layerState.seqLen !== currentSeqLen) {
    if (!layerState.warnedSeqMismatch) {
      layerState.warnedSeqMismatch = true;
      log.warn(
        'Layer',
        `linear_attention state mismatch at layer ${layerIdx}: state seqLen=${layerState.seqLen}, ` +
        `runtime seqLen=${currentSeqLen}. Resetting recurrent state.`
      );
    }
    clearDynamicLayerState(layerState);
    layerState.seqLen = currentSeqLen;
  }

  ensureLayerRuntimeGpuBuffers(layerState);
  return layerState;
}

async function projectLinearTensor({
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
  try {
    if (recorder) {
      return await recordMatmul(recorder, matmulInput, resolvedWeight, numTokens, outDim, hiddenSize, {
        transposeB: 'auto',
        role,
        layerIdx,
        kernelPath,
        outputDtype: resolvedOutputDtype,
        executionPolicies,
      });
    }
    return await runMatmul(matmulInput, resolvedWeight, numTokens, outDim, hiddenSize, {
      transposeB: 'auto',
      role,
      layerIdx,
      kernelPath,
      outputDtype: resolvedOutputDtype,
      executionPolicies,
    });
  } finally {
    if (matmulInput !== inputTensor) {
      releaseOrTrackBuffer(recorder, matmulInput.buffer);
    }
    releaseResolvedWeightBuffer(sourceWeight, resolvedWeight, recorder);
  }
}

export function hasLinearAttentionLayers(layerTypes) {
  if (!Array.isArray(layerTypes)) return false;
  for (let i = 0; i < layerTypes.length; i++) {
    const type = String(layerTypes[i] ?? '').trim().toLowerCase();
    if (
      type === 'linear_attention'
      || type === 'linear'
      || type === 'gated_delta'
      || type === 'gated_delta_net'
    ) {
      return true;
    }
  }
  return false;
}

export function createLinearAttentionRuntime() {
  log.debug(
    'Pipeline',
    'Linear attention runtime created (empty). Linear attention layers will be initialized on first use if model config declares them.'
  );
  return {
    schemaVersion: LINEAR_RUNTIME_SCHEMA_VERSION,
    layers: new Map(),
  };
}

export function resetLinearAttentionRuntime(runtime) {
  if (!runtime || typeof runtime !== 'object') {
    return createLinearAttentionRuntime();
  }
  releaseRuntimeLayerBuffers(runtime);
  runtime.schemaVersion = LINEAR_RUNTIME_SCHEMA_VERSION;
  runtime.layers = new Map();
  return runtime;
}

export async function cloneLinearAttentionRuntime(runtime) {
  if (!runtime || typeof runtime !== 'object' || !(runtime.layers instanceof Map)) {
    return createLinearAttentionRuntime();
  }

  const clonedLayers = new Map();
  for (const [layerIdx, layerState] of runtime.layers.entries()) {
    await syncLayerRuntimeStateFromGPU(layerState);
    clonedLayers.set(layerIdx, cloneLayerRuntimeState(layerState));
  }
  return {
    schemaVersion: LINEAR_RUNTIME_SCHEMA_VERSION,
    layers: clonedLayers,
  };
}

export function restoreLinearAttentionRuntime(runtime, snapshot) {
  const target = ensureRuntime(runtime);
  releaseRuntimeLayerBuffers(target);
  target.schemaVersion = LINEAR_RUNTIME_SCHEMA_VERSION;
  target.layers = new Map();
  if (!snapshot || typeof snapshot !== 'object') {
    return target;
  }
  if (snapshot.layers instanceof Map) {
    target.layers = cloneLayerMap(snapshot.layers);
  } else if (Array.isArray(snapshot.layers)) {
    for (const item of snapshot.layers) {
      if (!item || typeof item !== 'object' || !Number.isFinite(item.layerIdx)) {
        continue;
      }
      target.layers.set(Math.trunc(item.layerIdx), cloneLayerRuntimeState(item));
    }
  }
  return target;
}

export async function runLinearAttentionLayer(inputTensor, layerWeights, options) {
  const {
    layerIdx,
    numTokens,
    hiddenSize,
    config,
    currentSeqLen,
    activationDtype,
    kernelPath,
    linearRuntime,
    getWeightBuffer,
    getNormWeightBuffer,
    recorder,
    executionPolicies = null,
  } = options;

  if (!layerWeights) {
    throw new Error(`linear_attention layer ${layerIdx} has no weights.`);
  }
  if (!layerWeights.qkvProj || !layerWeights.oProj) {
    throw new Error(
      `linear_attention layer ${layerIdx} requires qkvProj and oProj weights.`
    );
  }
  if (!layerWeights.linearInProjZ || !layerWeights.linearInProjA || !layerWeights.linearInProjB) {
    throw new Error(
      `linear_attention layer ${layerIdx} requires in_proj_z, in_proj_a, and in_proj_b weights.`
    );
  }

  const runtime = ensureRuntime(linearRuntime);
  const projectionLayout = resolveProjectionLayout(config, layerWeights);
  const layerState = await getLayerRuntimeState(
    runtime,
    layerIdx,
    layerWeights,
    config,
    currentSeqLen,
    projectionLayout
  );

  const outputDtype = activationDtype === 'f16' ? 'f16' : 'f32';
  const projectionDtype = activationDtype === 'f16' ? 'f16' : 'f32';
  const phase = numTokens === 1 ? 'decode' : 'prefill';
  let normedTensor = inputTensor;
  let normedCreated = false;

  if (layerWeights.inputNorm) {
    const normWeightBuffer = getNormWeightBuffer(layerWeights.inputNorm, `L${layerIdx}.linear_input_norm`);
    try {
      if (recorder) {
        normedTensor = await recordRMSNorm(recorder, inputTensor, normWeightBuffer, layerState.rmsNormEps, {
          batchSize: numTokens,
          hiddenSize,
          rmsNormWeightOffset: config.rmsNormWeightOffset,
        });
      } else {
        normedTensor = await runRMSNorm(inputTensor, normWeightBuffer, layerState.rmsNormEps, {
          batchSize: numTokens,
          hiddenSize,
          rmsNormWeightOffset: config.rmsNormWeightOffset,
        });
      }
      normedCreated = true;
    } finally {
      if (!isGpuBuffer(layerWeights.inputNorm)) {
        releaseOrTrackBuffer(recorder, normWeightBuffer);
      }
    }
  }

  const qkvTensor = await projectLinearTensor({
    inputTensor: normedTensor,
    sourceWeight: layerWeights.qkvProj,
    role: 'linear_qkv_proj',
    phase,
    outDim: projectionLayout.convDim,
    numTokens,
    hiddenSize,
    layerIdx,
    kernelPath,
    outputDtype: projectionDtype,
    getWeightBuffer,
    recorder,
    executionPolicies,
  });
  const zTensor = await projectLinearTensor({
    inputTensor: normedTensor,
    sourceWeight: layerWeights.linearInProjZ,
    role: 'linear_z_proj',
    phase,
    outDim: projectionLayout.valueDim,
    numTokens,
    hiddenSize,
    layerIdx,
    kernelPath,
    outputDtype: projectionDtype,
    getWeightBuffer,
    recorder,
    executionPolicies,
  });
  const aTensor = await projectLinearTensor({
    inputTensor: normedTensor,
    sourceWeight: layerWeights.linearInProjA,
    role: 'linear_a_proj',
    phase,
    outDim: projectionLayout.numVHeads,
    numTokens,
    hiddenSize,
    layerIdx,
    kernelPath,
    outputDtype: projectionDtype,
    getWeightBuffer,
    recorder,
    executionPolicies,
  });
  const bTensor = await projectLinearTensor({
    inputTensor: normedTensor,
    sourceWeight: layerWeights.linearInProjB,
    role: 'linear_b_proj',
    phase,
    outDim: projectionLayout.numVHeads,
    numTokens,
    hiddenSize,
    layerIdx,
    kernelPath,
    outputDtype: projectionDtype,
    getWeightBuffer,
    recorder,
    executionPolicies,
  });

  const outProjInputDtype = resolveMatmulStepDtype(
    'linear_out_proj',
    phase,
    layerIdx,
    kernelPath,
    projectionDtype,
    'inputDtype'
  );
  const outProjOutputDtype = resolveMatmulStepDtype(
    'linear_out_proj',
    phase,
    layerIdx,
    kernelPath,
    outputDtype,
    'outputDtype'
  );

  try {
    await runProbes('linear_qkv_proj', qkvTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: projectionLayout.convDim,
      probes: options.debugProbes,
      recorder,
      operatorDiagnostics: options.operatorDiagnostics,
      dtype: qkvTensor.dtype,
    });
    await runProbes('linear_z_proj', zTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: projectionLayout.valueDim,
      probes: options.debugProbes,
      recorder,
      operatorDiagnostics: options.operatorDiagnostics,
      dtype: zTensor.dtype,
    });
    await runProbes('linear_a_proj', aTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: projectionLayout.numVHeads,
      probes: options.debugProbes,
      recorder,
      operatorDiagnostics: options.operatorDiagnostics,
      dtype: aTensor.dtype,
    });
    await runProbes('linear_b_proj', bTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: projectionLayout.numVHeads,
      probes: options.debugProbes,
      recorder,
      operatorDiagnostics: options.operatorDiagnostics,
      dtype: bTensor.dtype,
    });
    const coreTensor = await runLinearAttentionCoreGPU(
      qkvTensor,
      zTensor,
      aTensor,
      bTensor,
      layerState,
      {
        numTokens,
        outputDtype: outProjInputDtype,
        layerIdx,
        qkL2NormEps: QK_L2NORM_EPS,
        recorder,
        executionPolicies,
      }
    );
    await runProbes('linear_core_out', coreTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: projectionLayout.valueDim,
      probes: options.debugProbes,
      recorder,
      operatorDiagnostics: options.operatorDiagnostics,
      dtype: coreTensor.dtype,
    });
    layerState.seqLen = currentSeqLen + numTokens;
    const outProjWeight = getWeightBuffer(layerWeights.oProj, `L${layerIdx}.linear_out_proj`);
    try {
      let result;
      if (recorder) {
        result = await recordMatmul(recorder, coreTensor, outProjWeight, numTokens, hiddenSize, projectionLayout.valueDim, {
          transposeB: 'auto',
          role: 'linear_out_proj',
          layerIdx,
          kernelPath,
          outputDtype: outProjOutputDtype,
          executionPolicies,
        });
      } else {
        result = await runMatmul(coreTensor, outProjWeight, numTokens, hiddenSize, projectionLayout.valueDim, {
          transposeB: 'auto',
          role: 'linear_out_proj',
          layerIdx,
          kernelPath,
          outputDtype: outProjOutputDtype,
          executionPolicies,
        });
      }
      if (outputDtype === 'f16' && result.dtype !== 'f16') {
        assertImplicitDtypeTransitionAllowed({
          executionPolicies,
          fromDtype: result.dtype,
          toDtype: 'f16',
          op: 'linear_out_proj',
          detail: 'Linear attention output would narrow implicitly before leaving the layer.',
        });
        const casted = recorder
          ? await recordCastF32ToF16(recorder, result)
          : await castF32ToF16(result);
        releaseOrTrackBuffer(recorder, result.buffer);
        return casted;
      }
      return result;
    } finally {
      releaseOrTrackBuffer(recorder, coreTensor.buffer);
      releaseResolvedWeightBuffer(layerWeights.oProj, outProjWeight, recorder);
    }
  } finally {
    if (normedCreated) {
      releaseOrTrackBuffer(recorder, normedTensor.buffer);
    }
    releaseOrTrackBuffer(recorder, qkvTensor.buffer);
    releaseOrTrackBuffer(recorder, zTensor.buffer);
    releaseOrTrackBuffer(recorder, aTensor.buffer);
    releaseOrTrackBuffer(recorder, bTensor.buffer);
  }
}
