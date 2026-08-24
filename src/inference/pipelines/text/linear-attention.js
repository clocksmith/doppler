import { getBufferDtype, isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import {
  recordMatmul,
  recordRMSNorm,
  runMatmul,
  runRMSNorm,
  castF32ToF16,
  castF16ToF32,
  recordCastF32ToF16,
  recordCastF16ToF32,
} from '../../../gpu/kernel-selector.js';
import { readBuffer, releaseBuffer, uploadData, acquireBuffer } from '../../../memory/buffer-pool.js';
import { log } from '../../../debug/index.js';
import { decodeReadback, f16ToF32 } from './debug-utils/index.js';
import { runLinearAttentionCoreGPU } from '../../../gpu/kernels/linear-attention-core.js';
import {
  resolveLinearAttentionABProjection,
  resolveLinearAttentionQKVZProjection,
} from './linear-attention-ab-fusion.js';
import { runProbes } from './probes.js';
import { QK_K, Q4K_BLOCK_BYTES } from '../../../config/schema/index.js';
import { dequantizeQ4KM } from '../../../converter/quantizer.js';
import { getKernelPathMatmulPrecision } from '../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { assertImplicitDtypeTransitionAllowed } from './dtype-contract.js';
import { applyLoRA } from './lora-apply.js';
import { getLoRAModule } from './lora.js';
import { applyProjectionLoRA, createLayerRuntimeState, ensureLayerRuntimeGpuBuffers, inferLinearNormModeFromWeight, isGpuBuffer, normalizeLinearNormMode, projectLinearTensor, releaseOrTrackBuffer, releaseResolvedWeightBuffer, resolveMatmulStepDtype, toPositiveInt } from './linear-attention/plan.js';
export { applyLinearNormWeightOffset } from './linear-attention/plan.js';

const LINEAR_RUNTIME_SCHEMA_VERSION = 2;
const QK_L2NORM_EPS = 1e-6;

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
    aLog: layerState.aLog.slice(),
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

export function inferLinearNormMode(weight, projectionLayout) {
  return inferLinearNormModeFromWeight(weight, projectionLayout);
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
  if (isGpuBuffer(layerState.aLogGPU)) {
    releaseBuffer(layerState.aLogGPU);
    layerState.aLogGPU = null;
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

async function settleLinearAttentionCoreInputDtype(tensor, targetDtype, options) {
  if (tensor.dtype === targetDtype) {
    return tensor;
  }
  const {
    recorder,
    executionPolicies,
    role,
  } = options;
  assertImplicitDtypeTransitionAllowed({
    executionPolicies,
    fromDtype: tensor.dtype,
    toDtype: targetDtype,
    op: role,
    detail: 'Linear attention core requires qkv/z/a/b tensors to share one input dtype.',
  });
  if (targetDtype === 'f16') {
    return recorder ? await recordCastF32ToF16(recorder, tensor) : await castF32ToF16(tensor);
  }
  if (targetDtype === 'f32') {
    return recorder ? await recordCastF16ToF32(recorder, tensor) : await castF16ToF32(tensor);
  }
  throw new Error(`Unsupported linear_attention core input dtype "${String(targetDtype)}".`);
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
    precomputedInputNorm = null,
    lora = null,
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

  const projectionDtype = selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', {
    dtype: config?.inputDtype ?? activationDtype,
  });
  const layerOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', {
    dtype: config?.outputDtype ?? activationDtype,
  });
  const phase = numTokens === 1 ? 'decode' : 'prefill';
  let normedTensor = inputTensor;
  let normedCreated = false;

  if (precomputedInputNorm) {
    if (
      phase !== 'decode'
      || numTokens !== 1
      || !layerWeights.inputNorm
      || precomputedInputNorm.dtype !== inputTensor.dtype
      || precomputedInputNorm.dtype !== 'f32'
    ) {
      releaseOrTrackBuffer(recorder, precomputedInputNorm.buffer);
      throw new Error(`linear_attention layer ${layerIdx} received an incompatible precomputed input norm tensor.`);
    }
    normedTensor = precomputedInputNorm;
    normedCreated = true;
  } else if (layerWeights.inputNorm) {
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

  const qkvLoRA = getLoRAModule(lora, layerIdx, 'in_proj_qkv');
  const zLoRA = getLoRAModule(lora, layerIdx, 'in_proj_z');
  const aLoRA = getLoRAModule(lora, layerIdx, 'in_proj_a');
  const bLoRA = getLoRAModule(lora, layerIdx, 'in_proj_b');
  const outLoRA = getLoRAModule(lora, layerIdx, 'out_proj');
  const qkvzProjection = qkvLoRA || zLoRA ? null : resolveLinearAttentionQKVZProjection(layerWeights, {
    phase,
    numTokens,
    hiddenSize,
    numVHeads: projectionLayout.numVHeads,
    convDim: projectionLayout.convDim,
    valueDim: projectionLayout.valueDim,
    layerIdx,
    debugProbes: options.debugProbes,
    operatorDiagnostics: options.operatorDiagnostics,
  });
  const qkvzTensor = qkvzProjection
    ? await projectLinearTensor({
      inputTensor: normedTensor,
      sourceWeight: qkvzProjection.weight,
      role: 'linear_qkvz_proj',
      phase,
      outDim: qkvzProjection.outDim,
      numTokens,
      hiddenSize,
      layerIdx,
      kernelPath,
      outputDtype: projectionDtype,
      getWeightBuffer,
      recorder,
      executionPolicies,
    })
    : null;
  const qkvTensor = qkvzTensor ?? await projectLinearTensor({
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
    loraModule: qkvLoRA,
  });
  const zTensor = qkvzTensor ?? await projectLinearTensor({
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
    loraModule: zLoRA,
  });
  const abProjection = aLoRA || bLoRA ? null : resolveLinearAttentionABProjection(layerWeights, {
    phase,
    numTokens,
    hiddenSize,
    numVHeads: projectionLayout.numVHeads,
    layerIdx,
    debugProbes: options.debugProbes,
    operatorDiagnostics: options.operatorDiagnostics,
  });
  const abTensor = abProjection
    ? await projectLinearTensor({
      inputTensor: normedTensor,
      sourceWeight: abProjection.weight,
      role: 'linear_ab_proj',
      phase,
      outDim: abProjection.outDim,
      numTokens,
      hiddenSize,
      layerIdx,
      kernelPath,
      outputDtype: projectionDtype,
      getWeightBuffer,
      recorder,
      executionPolicies,
    })
    : null;
  const aTensor = abTensor ?? await projectLinearTensor({
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
    loraModule: aLoRA,
  });
  const bTensor = abTensor ?? await projectLinearTensor({
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
    loraModule: bLoRA,
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
    layerOutputDtype,
    'outputDtype'
  );
  let coreQkvTensor = qkvTensor;
  let coreZTensor = zTensor;
  let coreATensor = aTensor;
  let coreBTensor = bTensor;

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
    const coreInputDtype = qkvTensor.dtype;
    coreZTensor = await settleLinearAttentionCoreInputDtype(zTensor, coreInputDtype, {
      recorder,
      executionPolicies,
      role: 'linear_z_proj',
    });
    coreATensor = await settleLinearAttentionCoreInputDtype(aTensor, coreInputDtype, {
      recorder,
      executionPolicies,
      role: 'linear_a_proj',
    });
    coreBTensor = await settleLinearAttentionCoreInputDtype(bTensor, coreInputDtype, {
      recorder,
      executionPolicies,
      role: 'linear_b_proj',
    });
    const coreTensor = await runLinearAttentionCoreGPU(
      coreQkvTensor,
      coreZTensor,
      coreATensor,
      coreBTensor,
      layerState,
      {
        numTokens,
        outputDtype: outProjInputDtype,
        layerIdx,
        qkL2NormEps: QK_L2NORM_EPS,
        recorder,
        executionPolicies,
        abPacked: abProjection != null,
        qkvzPacked: qkvzProjection != null,
        bProjOffsetElements: abProjection?.bProjOffsetElements ?? 0,
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
      if (result.dtype !== outProjOutputDtype) {
        assertImplicitDtypeTransitionAllowed({
          executionPolicies,
          fromDtype: result.dtype,
          toDtype: outProjOutputDtype,
          op: 'linear_out_proj',
          detail: 'Linear attention output would change dtype implicitly before leaving the layer.',
        });
        const casted = outProjOutputDtype === 'f16'
          ? (recorder ? await recordCastF32ToF16(recorder, result) : await castF32ToF16(result))
          : (recorder ? await recordCastF16ToF32(recorder, result) : await castF16ToF32(result));
        releaseOrTrackBuffer(recorder, result.buffer);
        result = casted;
      }
      return await applyProjectionLoRA({
        inputTensor: coreTensor,
        baseOutput: result,
        loraModule: outLoRA,
        dims: { M: numTokens, N: hiddenSize, K: projectionLayout.valueDim },
        getWeightBuffer,
        recorder,
        kernelPath,
      });
    } finally {
      releaseOrTrackBuffer(recorder, coreTensor.buffer);
      releaseResolvedWeightBuffer(layerWeights.oProj, outProjWeight, recorder);
    }
  } finally {
    if (normedCreated) {
      releaseOrTrackBuffer(recorder, normedTensor.buffer);
    }
    const released = new Set();
    for (const tensor of [qkvTensor, zTensor, aTensor, bTensor, coreQkvTensor, coreZTensor, coreATensor, coreBTensor]) {
      const buffer = tensor?.buffer ?? null;
      if (!buffer || released.has(buffer)) continue;
      released.add(buffer);
      releaseOrTrackBuffer(recorder, buffer);
    }
  }
}
