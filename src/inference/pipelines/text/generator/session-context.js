import { mergeRuntimeValues } from '../../../../config/runtime-merge.js';
import {
  assertBundledAdapterAuthorized,
  assertKnownResolutionNotRevoked,
} from '../../../../config/revocation-policy.js';
import { log } from '../../../../debug/index.js';
import { allowReadback } from '../../../../gpu/perf-guards.js';
import { readBuffer } from '../../../../memory/buffer-pool.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { isPlainObject } from '../../../../formats/plain-object.js';
import { decodeReadback } from '../debug-utils/index.js';
import { resolveActiveExecutionPlan } from '../execution-plan.js';
import { resolveRangeAwareSelectiveWideningConfig } from '../finiteness-policy.js';

export function resolvePerLayerInputsSession(manifestSession, runtimeSession) {
  if (!isPlainObject(runtimeSession)) {
    return runtimeSession ?? (manifestSession ?? null);
  }
  if (!isPlainObject(manifestSession)) {
    return runtimeSession;
  }
  return mergeRuntimeValues(manifestSession, runtimeSession);
}

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
  const values = decodeReadback(data, dtype);
  let min = Infinity;
  let max = -Infinity;
  let nanCount = 0;
  let infCount = 0;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (Number.isNaN(value)) {
      nanCount++;
      continue;
    }
    if (!Number.isFinite(value)) {
      infCount++;
      continue;
    }
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const maxAbs = Number.isFinite(min) && Number.isFinite(max)
    ? Math.max(Math.abs(min), Math.abs(max))
    : Infinity;
  const sample = Array.from(values.slice(0, 6)).map((value) => value.toFixed(4)).join(', ');
  const expectedLabel = expectedDim ? ` expectedDim=${expectedDim}` : '';
  log.verbose(
    'Pipeline',
    `CHECK ${label}: dtype=${dtype} elems=${values.length}/${totalElements}${expectedLabel} ` +
    `min=${min.toFixed(4)} max=${max.toFixed(4)} maxAbs=${maxAbs.toFixed(4)} ` +
    `nan=${nanCount} inf=${infCount} sample=[${sample}]`
  );
}

function getWeightBufferConfig(state) {
  return {
    rmsNormWeightOffset: state.modelConfig.rmsNormWeightOffset,
  };
}

export function buildLayerContext(
  state,
  recorder,
  isDecodeMode,
  debugLayers,
  debugCheckBufferFn,
  executionPlan = null
) {
  assertKnownResolutionNotRevoked(state.revocationIdentity);
  assertBundledAdapterAuthorized(state.lora);
  const config = state.modelConfig;
  const computeConfig = state.runtimeConfig.inference.compute;
  const activeExecutionPlan = executionPlan ?? resolveActiveExecutionPlan(state);
  const activeKernelPath = activeExecutionPlan.kernelPath ?? state.resolvedKernelPath ?? null;
  const effectiveActivationDtype = activeExecutionPlan.activationDtype;
  const perLayerInputsSession = resolvePerLayerInputsSession(
    config.perLayerInputsSession ?? null,
    state.runtimeConfig?.inference?.session?.perLayerInputs ?? null
  );
  const effectiveComputeConfig = {
    ...computeConfig,
    activationDtype: effectiveActivationDtype,
  };
  const wideningPolicy = resolveRangeAwareSelectiveWideningConfig(computeConfig);
  const finitenessGuardEnabled = activeExecutionPlan.finitenessGuardEnabled;
  const finitenessAbsThreshold = activeExecutionPlan.finitenessAbsThreshold ?? wideningPolicy.absThreshold;
  const resolvedDebugLayers = debugLayers !== undefined
    ? debugLayers
    : state.runtimeConfig.shared.debug.pipeline.layers ?? null;
  const resolvedDebugFlags = state.debugFlags == null
    ? { debugLayers: resolvedDebugLayers }
    : {
      ...state.debugFlags,
      debugLayers: resolvedDebugLayers,
    };
  return {
    config,
    weights: state.weights,
    kvCache: state.kvCache,
    currentSeqLen: state.currentSeqLen,
    currentTokenIds: state.currentTokenIds ?? null,
    diffusionGemmaDecoder: false,
    __precomputedInputNorm: null,
    __postFfnNextInputNorm: null,
    useGPU: state.useGPU,
    debug: state.debug,
    stats: state.stats,
    ropeFreqsCos: state.ropeFreqsCos,
    ropeFreqsSin: state.ropeFreqsSin,
    ropeLocalCos: state.ropeLocalCos,
    ropeLocalSin: state.ropeLocalSin,
    sharedAttentionState: new Map(),
    linearAttentionRuntime: state.linearAttentionRuntime,
    convLayerStates: state.convLayerStates,
    weightConfig: getWeightBufferConfig(state),
    debugFlags: resolvedDebugFlags,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    debugCheckBuffer: debugCheckBufferFn,
    perLayerInputBuffer: null,
    perLayerInputsSession,
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
    resolvedRuntimeSession: state.resolvedRuntimeSession,
    observationContext: state.observationContext,
    executionPolicies: state.executionV1State?.policies ?? null,
    debugLayers: resolvedDebugLayers,
    finitenessBuffer: state.finitenessBuffer,
    finitenessGuardEnabled,
    finitenessAbsThreshold,
    step: state.decodeStepCount,
    phase: isDecodeMode ? 'decode' : 'prefill',
    operatorDiagnostics: state.operatorDiagnostics,
    useFusedGateUpGelu: state.runtimeConfig?.inference?.session?.useFusedGateUpGelu === true,
    useLargeBatchF16F32FusedGateUp: state.runtimeConfig?.inference?.session?.useLargeBatchF16F32FusedGateUp === true,
    useSandwichRMSNormPairFusion:
      state.runtimeConfig?.inference?.session?.useSandwichRMSNormPairFusion === true,
    usePostFfnNextInputRMSNormPairFusion:
      state.runtimeConfig?.inference?.session?.usePostFfnNextInputRMSNormPairFusion === true,
    skipKVCacheWrites: false,
  };
}
