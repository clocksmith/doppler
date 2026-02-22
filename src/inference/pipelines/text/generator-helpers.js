
import { readBuffer } from '../../../memory/buffer-pool.js';
import { allowReadback } from '../../../gpu/perf-guards.js';
import { log } from '../../../debug/index.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { decodeReadback } from './debug-utils.js';
import { isWeightBuffer, isCpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import { resolveRangeAwareSelectiveWideningConfig } from './finiteness-policy.js';


export async function debugCheckBuffer(state, buffer, label, numTokens, expectedDim) {
  if (!allowReadback(`pipeline.debug.${label}`)) return;

  const expectedElements = expectedDim ? numTokens * expectedDim : 0;
  let bytesPerElement = 4;
  if (expectedElements > 0) {
    const f16Bytes = expectedElements * 2;
    const f32Bytes = expectedElements * 4;
    if (buffer.size >= f32Bytes) {
      bytesPerElement = 4;
    } else if (buffer.size >= f16Bytes) {
      bytesPerElement = 2;
    } else {
      const rawBytes = buffer.size / expectedElements;
      if (Math.abs(rawBytes - 2) < 0.5) {
        bytesPerElement = 2;
      } else if (Math.abs(rawBytes - 4) < 0.5) {
        bytesPerElement = 4;
      } else {
        bytesPerElement = rawBytes < 3 ? 2 : 4;
      }
    }
  }

  const totalElements = expectedElements > 0
    ? expectedElements
    : Math.floor(buffer.size / bytesPerElement);
  const maxElements = Math.min(totalElements, 65536);
  const readBytes = Math.min(buffer.size, maxElements * bytesPerElement);

  const data = await readBuffer(buffer, readBytes);
  if (data.byteLength === 0) return;

  const dtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', { bytesPerElement });
  const arr = decodeReadback(data, dtype);

  let min = Infinity;
  let max = -Infinity;
  let nanCount = 0;
  let infCount = 0;

  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isNaN(v)) {
      nanCount++;
      continue;
    }
    if (!Number.isFinite(v)) {
      infCount++;
      continue;
    }
    if (v < min) min = v;
    if (v > max) max = v;
  }

  const maxAbs = Number.isFinite(min) && Number.isFinite(max)
    ? Math.max(Math.abs(min), Math.abs(max))
    : Infinity;
  const sample = Array.from(arr.slice(0, 6)).map(v => v.toFixed(4)).join(', ');
  const expectedLabel = expectedDim ? ` expectedDim=${expectedDim}` : '';

  log.verbose(
    'Pipeline',
    `CHECK ${label}: dtype=${dtype} elems=${arr.length}/${totalElements}${expectedLabel} ` +
    `min=${min.toFixed(4)} max=${max.toFixed(4)} maxAbs=${maxAbs.toFixed(4)} ` +
    `nan=${nanCount} inf=${infCount} sample=[${sample}]`
  );
}


export function buildLayerContext(state, recorder, isDecodeMode, debugLayers, debugCheckBufferFn) {
  const config = state.modelConfig;
  const computeConfig = state.runtimeConfig.inference.compute;
  const activeKernelPath = state.executionKernelPathOverride ?? state.resolvedKernelPath ?? null;
  const effectiveActivationDtype = state.executionActivationDtypeOverride ?? computeConfig.activationDtype;
  const effectiveComputeConfig = {
    ...computeConfig,
    activationDtype: effectiveActivationDtype,
  };
  const wideningPolicy = resolveRangeAwareSelectiveWideningConfig(computeConfig);

  const resolvedDebugLayers = debugLayers !== undefined
    ? debugLayers
    : state.runtimeConfig.shared.debug.pipeline.layers ?? null;

  return {
    config,
    weights: state.weights,
    kvCache: state.kvCache,
    currentSeqLen: state.currentSeqLen,
    currentTokenIds: state.currentTokenIds ?? null,
    useGPU: state.useGPU,
    debug: state.debug,
    stats: state.stats,
    ropeFreqsCos: state.ropeFreqsCos,
    ropeFreqsSin: state.ropeFreqsSin,
    ropeLocalCos: state.ropeLocalCos,
    ropeLocalSin: state.ropeLocalSin,
    weightConfig: getWeightBufferConfig(state),
    debugFlags: state.debugFlags,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    debugCheckBuffer: debugCheckBufferFn,
    pipelinePlan: state.layerPipelinePlan,
    expertWeights: state.expertWeights,
    expertLoader: state.dopplerLoader,
    moeRouter: state.moeRouter,
    layerRouterWeights: state.layerRouterWeights,
    recorder,
    lora: state.lora,
    decodeBuffers: isDecodeMode && state.decodeBuffers?.hasBuffers() ? state.decodeBuffers : null,
    runtimeComputeConfig: effectiveComputeConfig,
    activationDtype: effectiveActivationDtype,
    kernelPath: activeKernelPath,
    debugLayers: resolvedDebugLayers,
    finitenessBuffer: state.finitenessBuffer,
    finitenessGuardEnabled: wideningPolicy.enabled,
    finitenessAbsThreshold: wideningPolicy.absThreshold,
    step: state.decodeStepCount,
  };
}


export function getWeightBufferConfig(state) {
  return {
    rmsNormWeightOffset: state.modelConfig.rmsNormWeightOffset,
  };
}


export function getLogitsWeights(state) {
  const finalNorm = state.weights.get('final_norm');
  const lmHead = state.weights.get('lm_head');
  if (!finalNorm || !(finalNorm instanceof GPUBuffer || finalNorm instanceof Float32Array)) {
    throw new Error('Final norm not found or invalid type');
  }
  if (!lmHead || !(lmHead instanceof GPUBuffer || lmHead instanceof Float32Array || isWeightBuffer(lmHead) || isCpuWeightBuffer(lmHead))) {
    throw new Error('LM head not found or invalid type');
  }
  return { finalNorm, lmHead };
}


export function getLogitsConfig(state) {
  const config = state.modelConfig;
  const activeKernelPath = state.executionKernelPathOverride ?? state.resolvedKernelPath ?? null;
  const effectiveActivationDtype = state.executionActivationDtypeOverride
    ?? state.runtimeConfig.inference.compute.activationDtype;
  return {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    rmsNormEps: config.rmsNormEps,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
    useTiedEmbeddings: state.useTiedEmbeddings,
    embeddingVocabSize: state.embeddingVocabSize,
    finalLogitSoftcapping: config.finalLogitSoftcapping,
    largeWeights: state.runtimeConfig.inference.largeWeights,
    activationDtype: effectiveActivationDtype,
    kernelPath: activeKernelPath,
  };
}
