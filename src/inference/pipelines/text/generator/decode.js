import { getDevice, setTrackSubmits } from '../../../../gpu/device.js';
import { releaseBuffer, readBuffer } from '../../../../memory/buffer-pool.js';
import { recordArgmax, recordGPUSample, isGPUSamplingAvailable } from '../../../../gpu/kernels/sample.js';
import { recordRepPenalty } from '../../../../gpu/kernels/rep-penalty.js';
import { recordCheckStop } from '../../../../gpu/kernels/check-stop.js';
import { recordCheckHotVocabStop } from '../../../../gpu/kernels/check-hot-vocab-stop.js';
import { resetSubmitStats, logSubmitStats } from '../../../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../../../gpu/command-recorder.js';
import { allowReadback } from '../../../../gpu/perf-guards.js';
import { getUniformCache } from '../../../../gpu/uniform-cache.js';
import { log } from '../../../../debug/index.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import {
  isBatchDecodeEnabled,
  isDecodeRecorderEnabled,
  isProfileDecodeRecorderEnabled,
} from '../execution-plan.js';
import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from '../sampling.js';
import { isStopToken } from '../init.js';
import { embed } from '../embed.js';
import { resolvePerLayerInputsSession } from './session-context.js';
import { processLayer } from '../layer.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, recordGreedyLmHeadArgmaxGPU, extractLastPositionLogits, finalizeLogits, applySoftcapping } from '../logits/index.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer, getWeightDtype, getWeightMetadata } from '../../../../gpu/weight-buffer.js';
import { decodeReadback } from '../debug-utils/index.js';
import { captureObservedFusedDecodeLogits, emitObservedLogits } from '../generator-logits-observation.js';
import { getFinalNormWeights, extractEmbeddingFromHidden } from '../generator-runtime.js';
import { parseFinitenessStatusWords } from '../finiteness-guard-status.js';
import { hasLinearAttentionLayers } from '../linear-attention.js';
import { hasConvLayers } from '../layer.js';
import { advanceDecodeStepCount } from '../tsir-fixture-writer.js';
import {
  preparePerLayerInputs,
  createPleBufferCache,
  prefetchPerLayerRow,
  hasRangeBackedPerLayerInputEmbeddings,
  hasGpuSplitPerLayerInputEmbeddings,
  getPleHotVocabularyRuntime,
} from '../per-layer-inputs.js';

export const UNKNOWN_TOKEN_TEXT = '<unknown>';

export const FINITENESS_RESET_WORDS = new Uint32Array(4);

export function sumProfileTimings(timings) {
  if (!timings || Object.keys(timings).length === 0) return null;
  let total = 0;
  for (const value of Object.values(timings)) {
    if (Number.isFinite(value)) {
      total += value;
    }
  }
  return total;
}

export function getEffectiveActivationDtype(state, opts) {
  const executionPlanDtype = opts?.executionPlan?.activationDtype;
  if (executionPlanDtype !== undefined && executionPlanDtype !== null) {
    return executionPlanDtype;
  }
  if (executionPlanDtype === null) {
    throw new Error('[Pipeline] executionPlan.activationDtype is required when provided and cannot be null.');
  }
  return state.runtimeConfig.inference.compute.activationDtype;
}

export function getTokenTextOrUnknown(tokenizer, tokenId) {
  if (!tokenizer || typeof tokenizer.decode !== 'function') {
    return UNKNOWN_TOKEN_TEXT;
  }

  const tokenText = tokenizer.decode([tokenId], false, false);
  if (typeof tokenText !== 'string' || tokenText.length === 0) {
    return UNKNOWN_TOKEN_TEXT;
  }

  return tokenText;
}

export function isOwnedDecodeBuffer(candidate, decodeHiddenBuffer, decodeAltBuffer) {
  if (candidate === decodeHiddenBuffer) {
    return true;
  }
  return candidate === decodeAltBuffer;
}

export function releasePerLayerInputBuffer(buffer, recorder, decodeBuffers, pleCache = null) {
  if (!buffer) {
    return;
  }
  const ownsBuffer = decodeBuffers?.ownsBuffer(buffer) ?? false;
  if (ownsBuffer) {
    return;
  }
  const cachedPleBuffer = pleCache?.ownedBuffers instanceof Set && pleCache.ownedBuffers.has(buffer);
  if (cachedPleBuffer) {
    return;
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(buffer);
    return;
  }
  releaseBuffer(buffer);
}

export function schedulePlePrefetchForToken(state, tokenId) {
  if (state?.prefetchPleNextToken !== true) {
    return;
  }
  const config = state.modelConfig;
  const pleHiddenSize = Number(config?.hiddenSizePerLayerInput ?? 0);
  if (!Number.isFinite(pleHiddenSize) || pleHiddenSize <= 0) {
    return;
  }
  const pleWeights = state.weights.get('per_layer_inputs');
  if (!pleWeights?.embedTokensPerLayer) {
    return;
  }
  const resolvedPerLayerInputsSession = resolvePerLayerInputsSession(
    config.perLayerInputsSession ?? null,
    state.runtimeConfig?.inference?.session?.perLayerInputs ?? null
  );
  state.plePrefetchPending = prefetchPerLayerRow(
    tokenId,
    pleWeights.embedTokensPerLayer,
    config.numLayers * pleHiddenSize,
    resolvedPerLayerInputsSession
  );
}

export function getReusableSampleReadbackBuffer(state, device, size) {
  const existing = state.sampleReadbackBuffer;
  if (existing && existing.size >= size) {
    return existing;
  }
  if (existing) {
    existing.destroy();
  }
  const buffer = device.createBuffer({
    label: 'sample_staging_reuse',
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  state.sampleReadbackBuffer = buffer;
  return buffer;
}

export class FinitenessError extends Error {
  constructor(message) {
    super(message);
    this.name = 'FinitenessError';
  }
}

export function shouldLogProfileStep(state, step) {
  const profilerConfig = state.runtimeConfig?.shared?.debug?.profiler;
  const every = profilerConfig?.logEveryDecodeSteps ?? 1;
  if (!Number.isFinite(every) || every <= 1) return true;
  return step === 1 || step % every === 0;
}

export function recordDecodeProfileStep(state, entry) {
  if (!entry || (!entry.timings && !entry.recorderStats)) return;
  if (!state.stats.decodeProfileSteps) {
    state.stats.decodeProfileSteps = [];
  }
  state.stats.decodeProfileSteps.push(entry);
}

export function shouldUseFusedDecodeSampling(config) {
  return config.recorderEnabled === true
    && config.gpuSamplingEnabled === true
    && config.fusedDecodeDisabled !== true
    && !hasConvLayers(config.layerTypes ?? []);
}

export async function readSampledTokenFromStagingBuffer(stagingBuffer, options = {}) {
  const ownsStagingBuffer = options.ownsStagingBuffer === true;
  const hasFinitenessBuffer = options.hasFinitenessBuffer === true;
  const ring = options.ring ?? null;
  const cleanupRecorder = options.cleanupRecorder ?? null;
  const timing = {
    mapWaitMs: 0,
    cleanupMs: 0,
    copyMs: 0,
  };
  let mapped = false;
  let cleanupCompleted = false;

  try {
    const mapStart = performance.now();
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    timing.mapWaitMs = performance.now() - mapStart;
    mapped = true;
    const cleanupStart = performance.now();
    await cleanupRecorder?.completeDeferredCleanup();
    timing.cleanupMs = performance.now() - cleanupStart;
    cleanupCompleted = true;
    const copyStart = performance.now();
    const mappedWords = new Uint32Array(stagingBuffer.getMappedRange());
    const result = {
      nextToken: mappedWords[0],
      finitenessStatus: hasFinitenessBuffer
        ? parseFinitenessStatusWords(mappedWords, 1)
        : parseFinitenessStatusWords(mappedWords, 0),
      timing,
    };
    timing.copyMs = performance.now() - copyStart;
    return result;
  } finally {
    if (mapped) {
      stagingBuffer.unmap();
    }
    if (!cleanupCompleted) {
      await cleanupRecorder?.completeDeferredCleanup({ discardPooled: true });
    }
    if (ownsStagingBuffer) {
      stagingBuffer.destroy();
    }
    ring?.advance();
  }
}

export async function readMappedBufferCopy(stagingBuffer, options = {}) {
  const ownsStagingBuffer = options.ownsStagingBuffer !== false;
  let mapped = false;

  try {
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    mapped = true;
    return stagingBuffer.getMappedRange().slice(0);
  } finally {
    if (mapped) {
      stagingBuffer.unmap();
    }
    if (ownsStagingBuffer) {
      stagingBuffer.destroy();
    }
  }
}

export async function runDecodeLayers(state, tokenId, opts, helpers) {
  const config = state.modelConfig;
  const debugCheckBuffer = state.debug ? helpers.debugCheckBuffer : undefined;

  const context = helpers.buildLayerContext(undefined, true, opts.debugLayers, opts.executionPlan);
  context.currentTokenIds = [tokenId];

  state.decodeBuffers.resetPingPong();

  const decodeHiddenBuffer = state.decodeBuffers.getHiddenBuffer();
  const decodeAltBuffer = state.decodeBuffers.getOutputHiddenBuffer();

  const embedBufferRaw = state.weights.get('embed');
  if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !isSplitWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
    throw new Error('Embed buffer not found or not a supported buffer type');
  }
  const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
  const embedDtype = isCpuWeightBuffer(embedBufferRaw)
    ? embedBufferRaw.dtype
    : getWeightDtype(embedBufferRaw);
  const embedMetadata = getWeightMetadata(embedBufferRaw);
  const activationDtype = getEffectiveActivationDtype(state, opts);

  const embedTensor = await embed([tokenId], embedBuffer, {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    scaleEmbeddings: config.scaleEmbeddings,
    embeddingScale: config.embeddingScale, embeddingNormalization: config.embeddingNormalization,
    outputBuffer: decodeHiddenBuffer ?? undefined,
    transpose: state.embeddingTranspose,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    operatorDiagnostics: state.operatorDiagnostics,
    activationDtype,
    embeddingDtype: selectRuleValue('inference', 'dtype', 'embeddingDtype', { dtype: embedDtype }),
    embeddingStorageEncoding: embedMetadata?.storageEncoding ?? null,
    executionPolicies: state.executionV1State?.policies ?? null,
  });

  let hiddenStates = embedTensor.buffer;

  // Resolve pending PLE prefetch from previous decode step
  let plePrefetchResult = null;
  if (state.plePrefetchPending) {
    plePrefetchResult = await state.plePrefetchPending;
    state.plePrefetchPending = null;
  }

  const perLayerInputs = await preparePerLayerInputs([tokenId], embedTensor, context, {
    numTokens: 1,
    pleCache: state.pleCache ?? null,
    prefetchedRow: plePrefetchResult,
  });

  try {
    for (let l = 0; l < config.numLayers; l++) {
      context.perLayerInputBuffer = perLayerInputs?.[l] ?? null;
      const prevStates = hiddenStates;
      hiddenStates = (await processLayer(l, hiddenStates, 1, false, context));
      state.decodeBuffers.swapPingPong();
      releasePerLayerInputBuffer(context.perLayerInputBuffer, null, context.decodeBuffers, state.pleCache ?? null);
      if (perLayerInputs) {
        perLayerInputs[l] = null;
      }
      context.perLayerInputBuffer = null;
      if (isGpuBufferInstance(prevStates) && prevStates !== hiddenStates) {
        const isPreAllocated = isOwnedDecodeBuffer(prevStates, decodeHiddenBuffer, decodeAltBuffer);
        if (!isPreAllocated) {
          releaseBuffer(prevStates);
        }
      }
    }
  } finally {
    context.perLayerInputBuffer = null;
    if (perLayerInputs) {
      for (const buffer of perLayerInputs) {
        releasePerLayerInputBuffer(buffer, null, context.decodeBuffers, state.pleCache ?? null);
      }
    }
    helpers.releaseSharedAttentionState?.(context.sharedAttentionState, null);
  }

  return { hiddenStates, decodeHiddenBuffer, decodeAltBuffer, debugCheckBuffer, context };
}

export function createDecodeRecorder(state, opts) {
  const device = getDevice();
  const executionPlan = opts.executionPlan;
  const recorderConfig = {
    hasDevice: Boolean(device),
    debug: opts.debug,
    disableCommandBatching: executionPlan?.disableCommandBatching ?? opts.disableCommandBatching,
    kvLayout: state.kvCache?.layout ?? null,
  };
  const recorderEnabled = opts.profile
    ? isProfileDecodeRecorderEnabled(recorderConfig)
    : isDecodeRecorderEnabled(recorderConfig);
  let recorder;
  if (recorderEnabled) {
    const recorderOptions = {
      recordLabels: opts.debug === true || opts.benchmark === true || opts.executionObserver === true,
      recordDispatches: opts.debug === true || opts.executionObserver === true,
      aggregateDispatches: opts.executionObserver === true,
    };
    recorder = opts.profile
      ? createProfilingRecorder('decode', device, recorderOptions)
      : createCommandRecorder('decode', recorderOptions, device);
  }
  if (state.decodeStepCount === 1) {
    const path = selectRuleValue('inference', 'config', 'tracePath', { useRecorder: Boolean(recorder) });
    log.debug('Decode', `Using ${path} path (recorder=${!!recorder}, debug=${opts.debug})`);
  }
  return recorder;
}

export async function submitDecodeRecorderProfile(state, opts, recorder, profileLabel) {
  if (!recorder) {
    return;
  }
  await recorder.submitAndWait();

  if (!opts.profile) {
    return;
  }

  const timings = recorder.isProfilingEnabled()
    ? await recorder.resolveProfileTimings()
    : null;
  const total = sumProfileTimings(timings);
  if (total !== null) {
    state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
  }
  if (timings || opts.executionObserver) {
    recordDecodeProfileStep(state, {
      step: state.decodeStepCount,
      timings,
      totalMs: total ?? undefined,
      recorderStats: recorder.getStats(),
    });
    if (!opts.executionObserver && shouldLogProfileStep(state, state.decodeStepCount)) {
      log.warn('Profile', `Decode step ${state.decodeStepCount}${profileLabel}:`);
      log.warn('Profile', CommandRecorder.formatProfileReport(timings));
    }
  }
}

export async function decodeStep(state, currentIds, opts, helpers) {
  const stepWallStart = performance.now();
  const lastToken = currentIds[currentIds.length - 1];
  const numTokens = 1;
  const config = state.modelConfig;
  const samplingDefaults = state.runtimeConfig.inference.sampling;
  const executionPlan = opts.executionPlan;
  const debugCheckBuffer = state.debug ? helpers.debugCheckBuffer : undefined;

  advanceDecodeStepCount(state);
  const isDebugStep = opts.debug && state.decodeStepCount <= 5;
  if (isDebugStep) {
    const tokenText = getTokenTextOrUnknown(state.tokenizer, lastToken);
    log.debug('Decode', `[${state.decodeStepCount}] token="${tokenText}" pos=${state.currentSeqLen}`);
  }

  const device = getDevice();
  const recorder = createDecodeRecorder(state, opts);

  if (state.finitenessBuffer && device) {
    device.queue.writeBuffer(state.finitenessBuffer, 0, FINITENESS_RESET_WORDS);
  }

  const context = helpers.buildLayerContext(recorder, true, opts.debugLayers, executionPlan);
  context.currentTokenIds = [lastToken];

  state.decodeBuffers.resetPingPong();

  const decodeHiddenBuffer = state.decodeBuffers.getHiddenBuffer();
  const decodeAltBuffer = state.decodeBuffers.getOutputHiddenBuffer();

  const embedBufferRaw = state.weights.get('embed');
  if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !isSplitWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
    throw new Error('Embed buffer not found or not a supported buffer type');
  }
  const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
  const embedDtype = isCpuWeightBuffer(embedBufferRaw)
    ? embedBufferRaw.dtype
    : getWeightDtype(embedBufferRaw);
  const embedMetadata = getWeightMetadata(embedBufferRaw);
  const activationDtype = getEffectiveActivationDtype(state, opts);
  const activationBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });

  const embedTensor = await embed([lastToken], embedBuffer, {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    scaleEmbeddings: config.scaleEmbeddings,
    embeddingScale: config.embeddingScale, embeddingNormalization: config.embeddingNormalization,
    recorder,
    outputBuffer: decodeHiddenBuffer ?? undefined,
    transpose: state.embeddingTranspose,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    operatorDiagnostics: state.operatorDiagnostics,
    activationDtype,
    embeddingDtype: selectRuleValue('inference', 'dtype', 'embeddingDtype', { dtype: embedDtype }),
    embeddingStorageEncoding: embedMetadata?.storageEncoding ?? null,
    executionPolicies: state.executionV1State?.policies ?? null,
  });

  let hiddenStates = embedTensor.buffer;

  // Resolve pending PLE prefetch from previous decode step
  let plePrefetchResult = null;
  if (state.plePrefetchPending) {
    plePrefetchResult = await state.plePrefetchPending;
    state.plePrefetchPending = null;
  }

  const perLayerInputs = await preparePerLayerInputs([lastToken], embedTensor, context, {
    numTokens: 1,
    pleCache: state.pleCache ?? null,
    prefetchedRow: plePrefetchResult,
  });

  if (opts.debug && state.decodeStepCount === 1) {
    const validSize = config.hiddenSize * activationBytes;
    const embedData = await readBuffer(hiddenStates, validSize);
    const embedArr = decodeReadback(embedData, activationDtype);
    const sample = embedArr.slice(0, 5);
    const maxAbs = Math.max(...embedArr.map(Math.abs));
    const nonZero = embedArr.filter(x => Math.abs(x) > 1e-10).length;
    log.debug('Decode', `[1] Embed check: maxAbs=${maxAbs.toFixed(2)}, nonZero=${nonZero}/${embedArr.length}, sample=[${Array.from(sample).map(v => v.toFixed(3)).join(', ')}]`);
  }

  const benchmarkSubmits = state.decodeStepCount <= 3 && opts.debug;
  if (benchmarkSubmits) {
    setTrackSubmits(true);
    resetSubmitStats();
  }

  const hasGPUCache = context.kvCache?.hasGPUCache?.() ?? false;
  if (opts.debug && state.decodeStepCount === 1) {
    log.debug('Decode', `KV cache check: hasGPUCache=${hasGPUCache}, currentSeqLen=${context.currentSeqLen}`);
  }

  try {
    for (let l = 0; l < config.numLayers; l++) {
      context.perLayerInputBuffer = perLayerInputs?.[l] ?? null;
      const prevStates = hiddenStates;
      hiddenStates = (await processLayer(l, hiddenStates, numTokens, false, context));

      state.decodeBuffers.swapPingPong();
      releasePerLayerInputBuffer(
        context.perLayerInputBuffer,
        recorder,
        context.decodeBuffers,
        state.pleCache ?? null
      );
      if (perLayerInputs) {
        perLayerInputs[l] = null;
      }
      context.perLayerInputBuffer = null;

      if (isGpuBufferInstance(prevStates) && prevStates !== hiddenStates) {
        const isPreAllocated = isOwnedDecodeBuffer(prevStates, decodeHiddenBuffer, decodeAltBuffer);
        if (!isPreAllocated) {
          if (recorder) {
            recorder.trackTemporaryBuffer(prevStates);
          } else {
            releaseBuffer(prevStates);
          }
        }
      }
    }
  } finally {
    context.perLayerInputBuffer = null;
    if (perLayerInputs) {
      for (const buffer of perLayerInputs) {
        releasePerLayerInputBuffer(
          buffer,
          recorder,
          context.decodeBuffers,
          state.pleCache ?? null
        );
      }
    }
    helpers.releaseSharedAttentionState?.(context.sharedAttentionState, recorder);
  }

  const logitSoftcap = config.finalLogitSoftcapping === null
    ? 0
    : config.finalLogitSoftcapping;
  const padTokenId = state.tokenizer?.getSpecialTokens?.()?.pad ?? null;
  const lmHeadIsCpu = isCpuWeightBuffer(state.weights.get('lm_head'));
  const useGPUSampling = state.useGPU && isGPUSamplingAvailable() && !lmHeadIsCpu;
  const useFusedDecode = shouldUseFusedDecodeSampling({
    recorderEnabled: Boolean(recorder),
    gpuSamplingEnabled: useGPUSampling,
    fusedDecodeDisabled: state.disableFusedDecode,
    layerTypes: config.layerTypes,
  });

  if (useFusedDecode) {
    const ring = state.decodeRing;
    let ringSlot = null;
    if (ring) {
      ring.ensure({
        batchSize: 1,
        tokensPerInterval: 1,
        stopCheckMode: executionPlan?.stopCheckMode ?? opts.stopCheckMode,
        ringTokens: executionPlan?.ringTokens ?? state.runtimeConfig.inference.batching.ringTokens,
        ringStop: executionPlan?.ringStop ?? state.runtimeConfig.inference.batching.ringStop,
        ringStaging: executionPlan?.ringStaging ?? state.runtimeConfig.inference.batching.ringStaging,
      });
      ringSlot = ring.acquire();
    }

    const { logitsBuffer, vocabSize, logitsDtype } = await recordLogitsGPU(
      recorder,
      hiddenStates,
      numTokens,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig(),
      state.operatorDiagnostics,
    );

    const ringTokensBuffer = ringSlot?.tokens ?? null;
    const sampleOutputBuffer = opts.temperature < samplingDefaults.greedyThreshold
      ? await recordArgmax(recorder, logitsBuffer, vocabSize, {
        padTokenId,
        logitSoftcap,
        logitsDtype,
        outputBuffer: ringTokensBuffer ?? undefined,
        outputIndex: 0,
      })
      : await recordGPUSample(recorder, logitsBuffer, vocabSize, {
        temperature: opts.temperature,
        topK: opts.topK,
        padTokenId,
        logitSoftcap,
        logitsDtype,
        outputBuffer: ringTokensBuffer ?? undefined,
        outputIndex: 0,
        greedyThreshold: samplingDefaults.greedyThreshold,
        randomSeed: opts.seed,
      });

    const ringStagingBuffer = ringSlot?.stagingTokens ?? null;
    const stagingSize = state.finitenessBuffer ? 20 : 4;
    const stagingBuffer = ringStagingBuffer && ringStagingBuffer.size >= stagingSize
      ? ringStagingBuffer
      : getReusableSampleReadbackBuffer(state, device, stagingSize);
    const ownsStagingBuffer = false;
    const ownsSampleOutputBuffer = !ringTokensBuffer || sampleOutputBuffer !== ringTokensBuffer;

    const isPreAllocated = isOwnedDecodeBuffer(hiddenStates, decodeHiddenBuffer, decodeAltBuffer);
    const encoder = recorder.getEncoder();
    encoder.copyBufferToBuffer(sampleOutputBuffer, 0, stagingBuffer, 0, 4);
    if (state.finitenessBuffer) {
      encoder.copyBufferToBuffer(state.finitenessBuffer, 0, stagingBuffer, 4, 16);
    }

    const readbackMode = executionPlan?.readbackMode;
    const isOverlapped = readbackMode === 'overlapped';

    // In overlapped mode, advance ring BEFORE submit so the GPU's next copy
    // target is a fresh slot while we read the current one.
    if (isOverlapped) {
      ring?.advance();
    }

    if (!allowReadback('pipeline.decode.sample')) {
      throw new Error('[Pipeline] GPU readback disabled for sampling');
    }

    const submitStart = performance.now();
    recorder.submit({ cleanup: 'deferred' });
    const submitWaitMs = performance.now() - submitStart;

    const readbackStart = performance.now();
    const readbackResult = await readSampledTokenFromStagingBuffer(stagingBuffer, {
      ownsStagingBuffer,
      hasFinitenessBuffer: Boolean(state.finitenessBuffer),
      ring: isOverlapped ? null : ring,
      cleanupRecorder: recorder,
    });
    const readbackWaitMs = performance.now() - readbackStart;

    state.stats.singleTokenSubmitWaitMs = (state.stats.singleTokenSubmitWaitMs ?? 0) + submitWaitMs;
    state.stats.singleTokenReadbackWaitMs = (state.stats.singleTokenReadbackWaitMs ?? 0) + readbackWaitMs;
    if (readbackResult.timing) {
      state.stats.singleTokenReadbackMapWaitMs = (state.stats.singleTokenReadbackMapWaitMs ?? 0)
        + readbackResult.timing.mapWaitMs;
      state.stats.singleTokenReadbackCleanupMs = (state.stats.singleTokenReadbackCleanupMs ?? 0)
        + readbackResult.timing.cleanupMs;
      state.stats.singleTokenReadbackCopyMs = (state.stats.singleTokenReadbackCopyMs ?? 0)
        + readbackResult.timing.copyMs;
    }

    const { nextToken: fusedNextToken, finitenessStatus } = readbackResult;

    if (finitenessStatus.triggered) {
      releaseBuffer(logitsBuffer);
      if (ownsSampleOutputBuffer) releaseBuffer(sampleOutputBuffer);
      if (!isPreAllocated) releaseBuffer(hiddenStates);
      throw new FinitenessError(`F16 bounds exceeded during generation${finitenessStatus.metadata}`);
    }

    log.debug('Decode', `Step ${state.decodeStepCount}: token=${fusedNextToken} (vocabSize=${config.vocabSize})`);

    const invalidToken = fusedNextToken >= config.vocabSize
      || (padTokenId != null && fusedNextToken === padTokenId)
      || (padTokenId == null && fusedNextToken === 0);
    if (!invalidToken) await captureObservedFusedDecodeLogits(state, opts, logitsBuffer, vocabSize, logitsDtype, fusedNextToken, currentIds);
    if (!invalidToken) {
      schedulePlePrefetchForToken(state, fusedNextToken);
    }
    if (invalidToken) {
      log.warn('Decode', `Suspicious token ${fusedNextToken} (vocabSize=${config.vocabSize}, step=${state.decodeStepCount})`);
      if (allowReadback('pipeline.decode.debug-logits')) {
        const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsDtype });
        const logitSample = await readBuffer(logitsBuffer, Math.min(config.vocabSize * logitsBytes, 4096));
        const logitArr = decodeReadback(logitSample, logitsDtype);
        const maxLogit = Math.max(...logitArr);
        const minLogit = Math.min(...logitArr);
        const hasNaN = logitArr.some((v) => Number.isNaN(v));
        const hasInf = logitArr.some((v) => !Number.isFinite(v));
        let argmaxIdx = 0;
        let argmaxVal = logitArr[0];
        for (let i = 1; i < logitArr.length; i++) {
          if (logitArr[i] > argmaxVal) {
            argmaxVal = logitArr[i];
            argmaxIdx = i;
          }
        }
        log.warn('Decode', `Logits: max=${maxLogit.toFixed(4)} at [${argmaxIdx}], min=${minLogit.toFixed(4)}, hasNaN=${hasNaN}, hasInf=${hasInf}`);
        log.warn('Decode', `First 10 logits: ${Array.from(logitSample.slice(0, 10)).map((v) => v.toFixed(4)).join(', ')}`);
        log.warn('Decode', `Logit[0] (pad): ${logitArr[0].toFixed(4)}, Logit[${argmaxIdx}]: ${argmaxVal.toFixed(4)}`);
      }
    }

    releaseBuffer(logitsBuffer);
    if (ownsSampleOutputBuffer) {
      releaseBuffer(sampleOutputBuffer);
    }

    if (benchmarkSubmits) {
      logSubmitStats(`Decode step ${state.decodeStepCount} (${config.numLayers} layers, fused)`);
      setTrackSubmits(false);
    }

    if (opts.profile) {
      const timings = recorder.isProfilingEnabled()
        ? await recorder.resolveProfileTimings()
        : null;
      const total = sumProfileTimings(timings);
      if (total !== null) {
        state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
      }
      if (timings || opts.executionObserver) {
        recordDecodeProfileStep(state, {
          step: state.decodeStepCount,
          timings,
          totalMs: total ?? undefined,
          recorderStats: recorder.getStats(),
        });
        if (!opts.executionObserver && shouldLogProfileStep(state, state.decodeStepCount)) {
          log.warn('Profile', `Decode step ${state.decodeStepCount}:`);
          log.warn('Profile', CommandRecorder.formatProfileReport(timings));
        }
      }
    }

    if (invalidToken) {
      state.disableFusedDecode = true;
      log.warn('Decode', 'Fused sampling produced invalid token; falling back to CPU sampling.');
      const fallbackLogits = await computeLogits(
        hiddenStates,
        numTokens,
        helpers.getLogitsWeights(),
        helpers.getLogitsConfig(),
        state.useGPU,
        state.debugFlags,
        undefined,
        debugCheckBuffer,
        state.runtimeConfig.shared.debug.probes,
        null,
        state.operatorDiagnostics
      );
      applyRepetitionPenalty(fallbackLogits, currentIds, opts.repetitionPenalty);
      const fallbackToken = sample(fallbackLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
        padTokenId,
        seed: opts.seed,
      });
      schedulePlePrefetchForToken(state, fallbackToken);
      if (!isPreAllocated) {
        releaseBuffer(hiddenStates);
      }
      state.currentSeqLen++;
      return fallbackToken;
    }

    if (!isPreAllocated) {
      releaseBuffer(hiddenStates);
    }

    state.currentSeqLen++;
    const stepWallMs = performance.now() - stepWallStart;
    state.stats.singleTokenOrchestrationMs = (state.stats.singleTokenOrchestrationMs ?? 0)
      + Math.max(0, stepWallMs - submitWaitMs - readbackWaitMs);
    return fusedNextToken;
  }

  await submitDecodeRecorderProfile(state, opts, recorder, ' (layers only)');

  if (benchmarkSubmits) {
    logSubmitStats(`Decode step ${state.decodeStepCount} (${config.numLayers} layers)`);
    setTrackSubmits(false);
  }

  if (opts.debug && state.decodeStepCount === 1 && isGpuBufferInstance(hiddenStates)) {
    const debugDevice = getDevice();
    if (debugDevice) {
      if (allowReadback('pipeline.decode.debug-hidden')) {
        const debugReadbackSize = state.runtimeConfig.shared.debug.pipeline.readbackSampleSize;
        const sampleSize = Math.min(debugReadbackSize, hiddenStates.size);
        const staging = debugDevice.createBuffer({
          size: sampleSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc = debugDevice.createCommandEncoder();
        enc.copyBufferToBuffer(hiddenStates, 0, staging, 0, sampleSize);
        debugDevice.queue.submit([enc.finish()]);
        const data = new Float32Array(await readMappedBufferCopy(staging));
        const nanCount = Array.from(data).filter(x => !Number.isFinite(x)).length;
        const nonZero = Array.from(data).filter(x => Number.isFinite(x) && x !== 0).slice(0, 5);
        log.debug('Decode', `[1] HIDDEN_AFTER_LAYERS: nan=${nanCount}/${data.length}, nonZero=${nonZero.length}, sample=[${nonZero.map(x => x.toFixed(4)).join(', ')}]`);
      }
    }
  }

  if (useGPUSampling) {
    const logitsResult = await computeLogitsGPU(
      hiddenStates,
      numTokens,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig(),
      state.debugFlags,
      state.operatorDiagnostics
    );
    if (logitsResult) {
      const { logitsBuffer, vocabSize, logitsDtype } = logitsResult;
      const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsDtype });
      const nfReadbackStart = performance.now();
      const logitsData = await readBuffer(logitsBuffer, numTokens * vocabSize * logitsBytes);
      const nfReadbackMs = performance.now() - nfReadbackStart;
      state.stats.singleTokenReadbackWaitMs = (state.stats.singleTokenReadbackWaitMs ?? 0) + nfReadbackMs;
      releaseBuffer(logitsBuffer);

      const rawLogits = decodeReadback(logitsData, logitsDtype);
      const finalizedLogits = await finalizeLogits(
        rawLogits,
        numTokens,
        vocabSize,
        config.vocabSize,
        config,
        state.runtimeConfig.shared.debug.probes,
        state.operatorDiagnostics
      );
      const sampledLogits = extractLastPositionLogits(finalizedLogits, numTokens, config.vocabSize);

      applyRepetitionPenalty(sampledLogits, currentIds, opts.repetitionPenalty);
      const nextToken = sample(sampledLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
        padTokenId,
        seed: opts.seed,
      });
      emitObservedLogits(opts.onLogits, sampledLogits, nextToken, currentIds);
      if (!context.decodeBuffers?.ownsBuffer(hiddenStates)) {
        releaseBuffer(hiddenStates);
      }
      state.currentSeqLen++;
      const nfStepWallMs = performance.now() - stepWallStart;
      state.stats.singleTokenOrchestrationMs = (state.stats.singleTokenOrchestrationMs ?? 0)
        + Math.max(0, nfStepWallMs - nfReadbackMs);
      return nextToken;
    }
  }

  if (state.finitenessBuffer) {
    const isInfiniteData = await readBuffer(state.finitenessBuffer, 16);
    const u32 = new Uint32Array(isInfiniteData.buffer, isInfiniteData.byteOffset, 4);
    const finitenessStatus = parseFinitenessStatusWords(u32, 0);
    if (finitenessStatus.triggered) {
      if (!context.decodeBuffers?.ownsBuffer(hiddenStates)) {
        releaseBuffer(hiddenStates);
      }
      throw new FinitenessError(`F16 bounds exceeded during generation${finitenessStatus.metadata}`);
    }
  }

  const logits = await computeLogits(
    hiddenStates,
    numTokens,
    helpers.getLogitsWeights(),
    helpers.getLogitsConfig(),
    state.useGPU,
    state.debugFlags,
    undefined,
    debugCheckBuffer,
    state.runtimeConfig.shared.debug.probes,
    null,
    state.operatorDiagnostics
  );

  if (!context.decodeBuffers?.ownsBuffer(hiddenStates)) {
    releaseBuffer(hiddenStates);
  }

  if (isDebugStep) {
    logitsSanity(logits, `Decode[${state.decodeStepCount}]`, opts.decode);
  }

  applyRepetitionPenalty(logits, currentIds, opts.repetitionPenalty);
  const nextToken = sample(logits, {
    temperature: opts.temperature,
    topP: opts.topP,
    topK: opts.topK,
    padTokenId,
    seed: opts.seed,
  });

  state.currentSeqLen++;
  const cpuStepWallMs = performance.now() - stepWallStart;
  state.stats.singleTokenOrchestrationMs = (state.stats.singleTokenOrchestrationMs ?? 0) + cpuStepWallMs;
  return nextToken;
}

export async function decodeStepLogits(state, currentIds, opts, helpers) {
  const lastToken = currentIds[currentIds.length - 1];
  const numTokens = 1;
  const config = state.modelConfig;

  advanceDecodeStepCount(state);
  const recorder = createDecodeRecorder(state, opts);

  const { hiddenStates, decodeHiddenBuffer, decodeAltBuffer, debugCheckBuffer } = await runDecodeLayers(
    state,
    lastToken,
    opts,
    {
      ...helpers,
      buildLayerContext: (ignoredRecorder, isDecode, debugLayers, executionPlan) =>
        helpers.buildLayerContext(recorder, isDecode, debugLayers, executionPlan),
    }
  );

  await submitDecodeRecorderProfile(state, opts, recorder, ' (layers only)');

  let logitsBuffer = null;
  let logitsDtype = null;
  let rawVocabSize = config.vocabSize;
  let logits = null;

  if (state.useGPU && !isCpuWeightBuffer(state.weights.get('lm_head'))) {
    const logitsResult = await computeLogitsGPU(
      hiddenStates,
      numTokens,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig(),
      state.debugFlags,
      state.operatorDiagnostics
    );

    if (logitsResult) {
      logitsBuffer = logitsResult.logitsBuffer;
      logitsDtype = logitsResult.logitsDtype;
      rawVocabSize = logitsResult.vocabSize;

      const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsDtype });
      const logitsData = await readBuffer(logitsBuffer, numTokens * rawVocabSize * logitsBytes);
      const rawLogits = decodeReadback(logitsData, logitsDtype);
      const finalized = await finalizeLogits(
        rawLogits,
        numTokens,
        rawVocabSize,
        config.vocabSize,
        config,
        state.runtimeConfig.shared.debug.probes,
        state.operatorDiagnostics
      );
      logits = extractLastPositionLogits(finalized, numTokens, config.vocabSize);
    }
  }

  if (!logits) {
    const rawLogits = await computeLogits(
      hiddenStates,
      numTokens,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig(),
      state.useGPU,
      state.debugFlags,
      undefined,
      debugCheckBuffer,
      state.runtimeConfig.shared.debug.probes,
      null,
      state.operatorDiagnostics
    );
    logits = extractLastPositionLogits(rawLogits, numTokens, config.vocabSize);
  }

  const isPreAllocated = isOwnedDecodeBuffer(hiddenStates, decodeHiddenBuffer, decodeAltBuffer);
  if (!isPreAllocated) {
    releaseBuffer(hiddenStates);
  }

  state.currentSeqLen++;

  return {
    logits,
    logitsBuffer,
    logitsDtype,
    rawVocabSize,
    vocabSize: config.vocabSize,
  };
}
