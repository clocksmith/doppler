import { getBufferDtype, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { recordMatmul, recordRMSNorm, runMatmul, runRMSNorm } from '../../../gpu/kernel-selector.js';
import { readBuffer, releaseBuffer, uploadData, acquireBuffer } from '../../../memory/buffer-pool.js';
import { log } from '../../../debug/index.js';
import { decodeReadback } from './debug-utils.js';
import { runLinearAttentionCoreGPU } from '../../../gpu/kernels/linear-attention-core.js';

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
  return null;
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
  return configuredMode ?? inferredMode ?? 'shared';
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
  if (!elementCount) {
    const inferredBytes = sourceDtype === 'f16' || sourceDtype === 'bf16' ? 2 : 4;
    elementCount = Math.trunc(sourceBuffer.size / inferredBytes);
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
    convKernelSize = toPositiveInt(convKernel.shape[2]) ?? convKernelSize;
  }
  if (!convKernelSize) {
    convKernelSize = 4;
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

  const aNegExp = new Float32Array(aLog.length);
  for (let i = 0; i < aLog.length; i++) {
    aNegExp[i] = -Math.exp(aLog[i]);
  }

  const convState = new Float32Array(projectionLayout.convDim * convKernelSize);
  const recurrentState = new Float32Array(
    projectionLayout.numVHeads * projectionLayout.headKDim * projectionLayout.headVDim
  );
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
    rmsNormEps: Number(config.rmsNormEps) || 1e-6,
    convWeight,
    dtBias,
    aNegExp,
    normWeight: norm,
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
  outDim,
  numTokens,
  hiddenSize,
  layerIdx,
  kernelPath,
  outputDtype,
  getWeightBuffer,
  recorder,
}) {
  const resolvedWeight = getWeightBuffer(sourceWeight, role);
  try {
    if (recorder) {
      return await recordMatmul(recorder, inputTensor, resolvedWeight, numTokens, outDim, hiddenSize, {
        transposeB: 'auto',
        role,
        layerIdx,
        kernelPath,
        outputDtype,
      });
    }
    return await runMatmul(inputTensor, resolvedWeight, numTokens, outDim, hiddenSize, {
      transposeB: 'auto',
      role,
      layerIdx,
      kernelPath,
      outputDtype,
    });
  } finally {
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
  const projectionDtype = 'f32';
  let normedTensor = inputTensor;
  let normedCreated = false;

  if (layerWeights.inputNorm) {
    const normWeightBuffer = getNormWeightBuffer(layerWeights.inputNorm, `L${layerIdx}.linear_input_norm`);
    try {
      if (recorder) {
        normedTensor = await recordRMSNorm(recorder, inputTensor, normWeightBuffer, Number(config.rmsNormEps) || 1e-6, {
          batchSize: numTokens,
          hiddenSize,
          rmsNormWeightOffset: config.rmsNormWeightOffset,
        });
      } else {
        normedTensor = await runRMSNorm(inputTensor, normWeightBuffer, Number(config.rmsNormEps) || 1e-6, {
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
    outDim: projectionLayout.convDim,
    numTokens,
    hiddenSize,
    layerIdx,
    kernelPath,
    outputDtype: projectionDtype,
    getWeightBuffer,
    recorder,
  });
  const zTensor = await projectLinearTensor({
    inputTensor: normedTensor,
    sourceWeight: layerWeights.linearInProjZ,
    role: 'linear_z_proj',
    outDim: projectionLayout.valueDim,
    numTokens,
    hiddenSize,
    layerIdx,
    kernelPath,
    outputDtype: projectionDtype,
    getWeightBuffer,
    recorder,
  });
  const aTensor = await projectLinearTensor({
    inputTensor: normedTensor,
    sourceWeight: layerWeights.linearInProjA,
    role: 'linear_a_proj',
    outDim: projectionLayout.numVHeads,
    numTokens,
    hiddenSize,
    layerIdx,
    kernelPath,
    outputDtype: projectionDtype,
    getWeightBuffer,
    recorder,
  });
  const bTensor = await projectLinearTensor({
    inputTensor: normedTensor,
    sourceWeight: layerWeights.linearInProjB,
    role: 'linear_b_proj',
    outDim: projectionLayout.numVHeads,
    numTokens,
    hiddenSize,
    layerIdx,
    kernelPath,
    outputDtype: projectionDtype,
    getWeightBuffer,
    recorder,
  });

  try {
    const coreTensor = await runLinearAttentionCoreGPU(
      qkvTensor,
      zTensor,
      aTensor,
      bTensor,
      layerState,
      {
        numTokens,
        layerIdx,
        qkL2NormEps: QK_L2NORM_EPS,
        recorder,
      }
    );
    layerState.seqLen = currentSeqLen + numTokens;
    const outProjWeight = getWeightBuffer(layerWeights.oProj, `L${layerIdx}.linear_out_proj`);
    try {
      if (recorder) {
        return await recordMatmul(recorder, coreTensor, outProjWeight, numTokens, hiddenSize, projectionLayout.valueDim, {
          transposeB: 'auto',
          role: 'linear_out_proj',
          layerIdx,
          kernelPath,
          outputDtype,
        });
      }
      return await runMatmul(coreTensor, outProjWeight, numTokens, hiddenSize, projectionLayout.valueDim, {
        transposeB: 'auto',
        role: 'linear_out_proj',
        layerIdx,
        kernelPath,
        outputDtype,
      });
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
