import { getDevice, setTrackSubmits } from '../../../gpu/device.js';
import { releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { recordArgmax, recordGPUSample, isGPUSamplingAvailable } from '../../../gpu/kernels/sample.js';
import { recordRepPenalty } from '../../../gpu/kernels/rep-penalty.js';
import { recordCheckStop } from '../../../gpu/kernels/check-stop.js';
import { recordCheckHotVocabStop } from '../../../gpu/kernels/check-hot-vocab-stop.js';
import { resetSubmitStats, logSubmitStats } from '../../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../../gpu/command-recorder.js';
import { allowReadback } from '../../../gpu/perf-guards.js';
import { getUniformCache } from '../../../gpu/uniform-cache.js';
import { log } from '../../../debug/index.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import {
  isBatchDecodeEnabled,
  isDecodeRecorderEnabled,
  isProfileDecodeRecorderEnabled,
} from './execution-plan.js';

import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from './sampling.js';
import { isStopToken } from './init.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, extractLastPositionLogits, finalizeLogits, applySoftcapping } from './logits/index.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, getWeightDtype } from '../../../gpu/weight-buffer.js';
import { decodeReadback } from './debug-utils/index.js';
import { getFinalNormWeights, extractEmbeddingFromHidden } from './generator-runtime.js';
import { parseFinitenessStatusWords } from './finiteness-guard-status.js';
import { hasLinearAttentionLayers } from './linear-attention.js';
import { hasConvLayers } from './layer.js';
import {
  preparePerLayerInputs,
  createPleBufferCache,
  prefetchPerLayerRow,
  hasRangeBackedPerLayerInputEmbeddings,
  getPleHotVocabularyRuntime,
} from './per-layer-inputs.js';

const UNKNOWN_TOKEN_TEXT = '<unknown>';

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

function getEffectiveActivationDtype(state, opts) {
  const executionPlanDtype = opts?.executionPlan?.activationDtype;
  if (executionPlanDtype !== undefined && executionPlanDtype !== null) {
    return executionPlanDtype;
  }
  if (executionPlanDtype === null) {
    throw new Error('[Pipeline] executionPlan.activationDtype is required when provided and cannot be null.');
  }
  return state.runtimeConfig.inference.compute.activationDtype;
}

function getTokenTextOrUnknown(tokenizer, tokenId) {
  if (!tokenizer || typeof tokenizer.decode !== 'function') {
    return UNKNOWN_TOKEN_TEXT;
  }

  const tokenText = tokenizer.decode([tokenId], false, false);
  if (typeof tokenText !== 'string' || tokenText.length === 0) {
    return UNKNOWN_TOKEN_TEXT;
  }

  return tokenText;
}

function isOwnedDecodeBuffer(candidate, decodeHiddenBuffer, decodeAltBuffer) {
  if (candidate === decodeHiddenBuffer) {
    return true;
  }
  return candidate === decodeAltBuffer;
}

function releasePerLayerInputBuffer(buffer, recorder, decodeBuffers, pleCache = null) {
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

function schedulePlePrefetchForToken(state, tokenId) {
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
  const perLayerInputsSession = state.runtimeConfig.inference.session?.perLayerInputs ?? config.perLayerInputsSession;
  state.plePrefetchPending = prefetchPerLayerRow(
    tokenId,
    pleWeights.embedTokensPerLayer,
    config.numLayers * pleHiddenSize,
    perLayerInputsSession
  );
}

function getReusableSampleReadbackBuffer(state, device, size) {
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

function shouldLogProfileStep(state, step) {
  const profilerConfig = state.runtimeConfig?.shared?.debug?.profiler;
  const every = profilerConfig?.logEveryDecodeSteps ?? 1;
  if (!Number.isFinite(every) || every <= 1) return true;
  return step === 1 || step % every === 0;
}

function recordDecodeProfileStep(state, entry) {
  if (!entry || !entry.timings) return;
  if (!state.stats.decodeProfileSteps) {
    state.stats.decodeProfileSteps = [];
  }
  state.stats.decodeProfileSteps.push(entry);
}

export function shouldUseBatchDecode(config) {
  return isBatchDecodeEnabled(config);
}

export function shouldUseFusedDecodeSampling(config) {
  return config.recorderEnabled === true
    && config.gpuSamplingEnabled === true
    && config.fusedDecodeDisabled !== true
    && !hasConvLayers(config.layerTypes ?? []);
}

export function resolveBatchStop(tokens, stopFlags, stopTokenIds, eosTokenId) {
  let actualCount = tokens.length;
  if (stopFlags) {
    const maxFlags = Math.min(stopFlags.length, tokens.length);
    for (let i = 0; i < maxFlags; i++) {
      if (stopFlags[i] === 1) {
        actualCount = i + 1;
        break;
      }
    }
  }

  for (let i = 0; i < actualCount; i++) {
    if (isStopToken(tokens[i], stopTokenIds, eosTokenId)) {
      actualCount = i + 1;
      break;
    }
  }

  return actualCount;
}

export function findInvalidGeneratedToken(tokens, vocabSize, padTokenId = null) {
  for (let i = 0; i < tokens.length; i++) {
    const tokenId = tokens[i];
    const isInvalid = !Number.isFinite(tokenId)
      || tokenId < 0
      || tokenId >= vocabSize
      || (padTokenId != null ? tokenId === padTokenId : tokenId === 0);
    if (isInvalid) {
      return { index: i, tokenId };
    }
  }
  return null;
}

export async function readSampledTokenFromStagingBuffer(stagingBuffer, options = {}) {
  const ownsStagingBuffer = options.ownsStagingBuffer === true;
  const hasFinitenessBuffer = options.hasFinitenessBuffer === true;
  const ring = options.ring ?? null;
  const cleanupRecorder = options.cleanupRecorder ?? null;
  let mapped = false;
  let cleanupCompleted = false;

  try {
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    mapped = true;
    await cleanupRecorder?.completeDeferredCleanup();
    cleanupCompleted = true;
    const mappedWords = new Uint32Array(stagingBuffer.getMappedRange());
    return {
      nextToken: mappedWords[0],
      finitenessStatus: hasFinitenessBuffer
        ? parseFinitenessStatusWords(mappedWords, 1)
        : parseFinitenessStatusWords(mappedWords, 0),
    };
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

export async function readBatchTokensFromStagingBuffers(options) {
  const {
    tokensStagingBuffer,
    stopStagingBuffer = null,
    finitenessStagingBuffer = null,
    tokenCount,
    ownsTokensStaging = false,
    ownsStopStaging = false,
    ownsFinitenessStaging = Boolean(finitenessStagingBuffer),
    ring = null,
    cleanupRecorder = null,
  } = options;
  let tokensMapped = false;
  let stopMapped = false;
  let finitenessMapped = false;
  let cleanupCompleted = false;

  try {
    const mapPromises = [tokensStagingBuffer.mapAsync(GPUMapMode.READ)];
    if (stopStagingBuffer) {
      mapPromises.push(stopStagingBuffer.mapAsync(GPUMapMode.READ));
    }
    if (finitenessStagingBuffer) {
      mapPromises.push(finitenessStagingBuffer.mapAsync(GPUMapMode.READ));
    }
    const mapResults = await Promise.allSettled(mapPromises);
    tokensMapped = mapResults[0]?.status === 'fulfilled';
    stopMapped = Boolean(stopStagingBuffer) && mapResults[1]?.status === 'fulfilled';
    finitenessMapped = Boolean(finitenessStagingBuffer)
      && mapResults[stopStagingBuffer ? 2 : 1]?.status === 'fulfilled';
    const mapFailure = mapResults.find((result) => result.status === 'rejected');
    if (mapFailure) {
      throw mapFailure.reason;
    }
    await cleanupRecorder?.completeDeferredCleanup();
    cleanupCompleted = true;

    const tokens = Array.from(
      new Uint32Array(tokensStagingBuffer.getMappedRange()).subarray(0, tokenCount)
    );
    const stopFlags = stopStagingBuffer
      ? new Uint32Array(stopStagingBuffer.getMappedRange().slice(0, tokenCount * 4))
      : null;
    const finitenessStatus = finitenessStagingBuffer
      ? parseFinitenessStatusWords(new Uint32Array(finitenessStagingBuffer.getMappedRange()), 0)
      : { triggered: false, metadata: '' };

    return {
      tokens,
      stopFlags,
      finitenessStatus,
    };
  } finally {
    if (finitenessMapped) {
      finitenessStagingBuffer.unmap();
    }
    if (tokensMapped) {
      tokensStagingBuffer.unmap();
    }
    if (stopMapped) {
      stopStagingBuffer.unmap();
    }
    if (!cleanupCompleted) {
      await cleanupRecorder?.completeDeferredCleanup({ discardPooled: true });
    }
    if (ownsFinitenessStaging) {
      finitenessStagingBuffer.destroy();
    }
    if (ownsTokensStaging) {
      tokensStagingBuffer.destroy();
    }
    if (ownsStopStaging) {
      stopStagingBuffer?.destroy();
    }
    ring?.advance();
  }
}

async function runDecodeLayers(state, tokenId, opts, helpers) {
  const config = state.modelConfig;
  const debugCheckBuffer = state.debug ? helpers.debugCheckBuffer : undefined;

  const context = helpers.buildLayerContext(undefined, true, opts.debugLayers, opts.executionPlan);
  context.currentTokenIds = [tokenId];

  state.decodeBuffers.resetPingPong();

  const decodeHiddenBuffer = state.decodeBuffers.getHiddenBuffer();
  const decodeAltBuffer = state.decodeBuffers.getOutputHiddenBuffer();

  const embedBufferRaw = state.weights.get('embed');
  if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
    throw new Error('Embed buffer not found or not a supported buffer type');
  }
  const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
  const embedDtype = isCpuWeightBuffer(embedBufferRaw)
    ? embedBufferRaw.dtype
    : getWeightDtype(embedBufferRaw);
  const activationDtype = getEffectiveActivationDtype(state, opts);

  const embedTensor = await embed([tokenId], embedBuffer, {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    scaleEmbeddings: config.scaleEmbeddings,
    outputBuffer: decodeHiddenBuffer ?? undefined,
    transpose: state.embeddingTranspose,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    operatorDiagnostics: state.operatorDiagnostics,
    activationDtype,
    embeddingDtype: selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype: embedDtype }),
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

function createDecodeRecorder(state, opts) {
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
    recorder = opts.profile
      ? createProfilingRecorder('decode')
      : createCommandRecorder('decode');
  }
  if (state.decodeStepCount === 1) {
    const path = selectRuleValue('inference', 'config', 'tracePath', { useRecorder: Boolean(recorder) });
    log.debug('Decode', `Using ${path} path (recorder=${!!recorder}, debug=${opts.debug})`);
  }
  return recorder;
}

async function submitDecodeRecorderProfile(state, opts, recorder, profileLabel) {
  if (!recorder) {
    return;
  }
  await recorder.submitAndWait();

  if (!opts.profile || !recorder.isProfilingEnabled()) {
    return;
  }

  const timings = await recorder.resolveProfileTimings();
  const total = sumProfileTimings(timings);
  if (total !== null) {
    state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
  }
  if (timings) {
    recordDecodeProfileStep(state, { step: state.decodeStepCount, timings, totalMs: total ?? undefined });
    if (shouldLogProfileStep(state, state.decodeStepCount)) {
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

  state.decodeStepCount++;
  const isDebugStep = opts.debug && state.decodeStepCount <= 5;
  if (isDebugStep) {
    const tokenText = getTokenTextOrUnknown(state.tokenizer, lastToken);
    log.debug('Decode', `[${state.decodeStepCount}] token="${tokenText}" pos=${state.currentSeqLen}`);
  }

  const device = getDevice();
  const recorder = createDecodeRecorder(state, opts);

  if (state.finitenessBuffer && device) {
    device.queue.writeBuffer(state.finitenessBuffer, 0, new Uint32Array([0, 0, 0, 0]));
  }

  const context = helpers.buildLayerContext(recorder, true, opts.debugLayers, executionPlan);
  context.currentTokenIds = [lastToken];

  state.decodeBuffers.resetPingPong();

  const decodeHiddenBuffer = state.decodeBuffers.getHiddenBuffer();
  const decodeAltBuffer = state.decodeBuffers.getOutputHiddenBuffer();

  const embedBufferRaw = state.weights.get('embed');
  if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
    throw new Error('Embed buffer not found or not a supported buffer type');
  }
  const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
  const embedDtype = isCpuWeightBuffer(embedBufferRaw)
    ? embedBufferRaw.dtype
    : getWeightDtype(embedBufferRaw);
  const activationDtype = getEffectiveActivationDtype(state, opts);
  const activationBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });

  const embedTensor = await embed([lastToken], embedBuffer, {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    scaleEmbeddings: config.scaleEmbeddings,
    recorder,
    outputBuffer: decodeHiddenBuffer ?? undefined,
    transpose: state.embeddingTranspose,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    operatorDiagnostics: state.operatorDiagnostics,
    activationDtype,
    embeddingDtype: selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype: embedDtype }),
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

    if (opts.profile && recorder.isProfilingEnabled()) {
      const timings = await recorder.resolveProfileTimings();
      const total = sumProfileTimings(timings);
      if (total !== null) {
        state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
      }
      if (timings) {
        recordDecodeProfileStep(state, { step: state.decodeStepCount, timings, totalMs: total ?? undefined });
        if (shouldLogProfileStep(state, state.decodeStepCount)) {
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

  state.decodeStepCount++;
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

export async function advanceWithToken(state, tokenId, opts, helpers) {
  state.decodeStepCount++;

  const { hiddenStates, decodeHiddenBuffer, decodeAltBuffer } = await runDecodeLayers(
    state,
    tokenId,
    opts,
    helpers
  );

  const isPreAllocated = isOwnedDecodeBuffer(hiddenStates, decodeHiddenBuffer, decodeAltBuffer);
  if (!isPreAllocated) {
    releaseBuffer(hiddenStates);
  }

  state.currentSeqLen++;
}

export async function advanceWithTokenAndEmbedding(state, tokenId, opts, helpers, embeddingMode) {

  state.decodeStepCount++;

  const { hiddenStates, decodeHiddenBuffer, decodeAltBuffer } = await runDecodeLayers(
    state,
    tokenId,
    opts,
    helpers
  );

  if (!allowReadback('pipeline.advance.embedding')) {
    throw new Error('GPU readback disabled; cannot return embedding');
  }

  const device = getDevice();
  if (!device) {
    throw new Error('GPU device not available');
  }

  const config = state.modelConfig;
  const activationDtype = getEffectiveActivationDtype(state, opts);
  const activationBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });

  let embedding;
  try {
    const sampleSize = config.hiddenSize * activationBytes;
    const staging = device.createBuffer({
      size: sampleSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    let decodedHidden;
    let stagingMapped = false;
    try {
      const enc = device.createCommandEncoder({ label: 'advance_with_embedding_copy' });
      enc.copyBufferToBuffer(hiddenStates, 0, staging, 0, sampleSize);
      device.queue.submit([enc.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      stagingMapped = true;
      decodedHidden = decodeReadback(staging.getMappedRange().slice(0), activationDtype);
    } finally {
      if (stagingMapped) {
        staging.unmap();
      }
      staging.destroy();
    }
    const finalNormWeights = await getFinalNormWeights(state);
    embedding = extractEmbeddingFromHidden(
      decodedHidden,
      1,
      config.hiddenSize,
      embeddingMode,
      finalNormWeights,
      config,
      state.embeddingPostprocessor
    );
  } finally {
    const isPreAllocated = isOwnedDecodeBuffer(hiddenStates, decodeHiddenBuffer, decodeAltBuffer);
    if (!isPreAllocated) {
      releaseBuffer(hiddenStates);
    }
  }

  state.currentSeqLen++;

  return {
    embedding,
    embeddingMode,
    seqLen: state.currentSeqLen,
  };
}

async function generateNTokensGPUStepwiseRangeBackedPle(state, N, currentIds, opts, helpers) {
  const config = state.modelConfig;
  const batchStart = performance.now();
  state.batchingStats.batchedForwardCalls += 1;

  const stopTokenIds = config.stopTokenIds;
  const eosToken = state.tokenizer?.getSpecialTokens?.()?.eos;
  const pleHiddenSize = Number(config.hiddenSizePerLayerInput ?? 0);
  const totalPerLayerHiddenSize = pleHiddenSize > 0
    ? config.numLayers * pleHiddenSize
    : 0;
  const pleWeights = state.weights.get('per_layer_inputs');
  const pleSession = state.runtimeConfig.inference.session?.perLayerInputs ?? config.perLayerInputsSession;
  const generatedTokens = [];
  const rollingIds = Array.isArray(currentIds) ? currentIds.slice() : Array.from(currentIds ?? []);
  let gpuSubmissions = 0;

  try {
    state.prefetchPleNextToken = true;
    if (
      totalPerLayerHiddenSize > 0
      && pleWeights?.embedTokensPerLayer
      && !state.plePrefetchPending
      && rollingIds.length > 0
    ) {
      state.plePrefetchPending = prefetchPerLayerRow(
        rollingIds[rollingIds.length - 1],
        pleWeights.embedTokensPerLayer,
        totalPerLayerHiddenSize,
        pleSession
      );
    }

    for (let i = 0; i < N; i += 1) {
      const nextToken = await decodeStep(state, rollingIds, opts, helpers);
      gpuSubmissions += 1;
      generatedTokens.push(nextToken);
      rollingIds.push(nextToken);

      if (isStopToken(nextToken, stopTokenIds, eosToken)) {
        break;
      }
    }

    return {
      tokens: generatedTokens,
      actualCount: generatedTokens.length,
    };
  } finally {
    state.prefetchPleNextToken = false;
    state.batchingStats.totalBatchedTimeMs += Math.max(0, performance.now() - batchStart);
    state.batchingStats.gpuSubmissions += gpuSubmissions;
  }
}

export async function generateNTokensGPU(state, startToken, N, currentIds, opts, helpers) {
  const device = getDevice();
  const config = state.modelConfig;
  if (hasConvLayers(config.layerTypes)) {
    throw new Error(
      '[Pipeline] Batch decode path is disabled for conv models; use single-token decode.'
    );
  }
  const samplingDefaults = state.runtimeConfig.inference.sampling;
  const executionPlan = opts.executionPlan;
  const batchSize = executionPlan?.batchSize ?? opts.batchSize ?? state.runtimeConfig.inference.batching.batchSize;
  const readbackIntervalRaw = executionPlan?.readbackInterval ?? state.runtimeConfig.inference.batching.readbackInterval;
  const readbackInterval = readbackIntervalRaw == null ? 1 : readbackIntervalRaw;
  const stopCheckMode = executionPlan?.stopCheckMode ?? opts.stopCheckMode ?? state.runtimeConfig.inference.batching.stopCheckMode;
  // GPU stop-flag checks are only useful when we read back every token.
  // With deferred readback, we already scan sampled tokens on CPU to find the
  // earliest stop token, so extra stop buffers/kernels are redundant overhead.
  let useGpuStopFlags = stopCheckMode === 'per-token' && readbackInterval <= 1;
  let effectiveStopCheckMode = useGpuStopFlags ? 'per-token' : 'batch';
  const batchStart = performance.now();

  state.batchingStats.batchedForwardCalls += 1;
  const tokensPerInterval = batchSize * readbackInterval;
  const recorder = opts.profile
    ? createProfilingRecorder('batch_decode')
    : createCommandRecorder('batch_decode');
  const lmHead = state.weights.get('lm_head');
  if (lmHead && isCpuWeightBuffer(lmHead)) {
    throw new Error('[Pipeline] GPU-only decode not supported with CPU-resident LM head.');
  }

  if (!Number.isFinite(N) || N <= 0) {
    throw new Error('[Pipeline] generateNTokensGPU requires N > 0.');
  }
  if (N > tokensPerInterval) {
    throw new Error('[Pipeline] Batch size exceeds decode ring capacity.');
  }

  const hasRangeBackedPerLayerInputs = hasRangeBackedPerLayerInputEmbeddings({
    config,
    weights: state.weights,
  });
  const pleHotVocabularyRuntime = getPleHotVocabularyRuntime({ weights: state.weights });
  const hotStartTokenIndex = pleHotVocabularyRuntime?.hotTokenIndexMap?.[startToken] ?? null;
  const canUseHotVocabularyBatchDecode = hasRangeBackedPerLayerInputs
    && pleHotVocabularyRuntime
    && hotStartTokenIndex != null
    && hotStartTokenIndex !== pleHotVocabularyRuntime.sentinelIndex;
  if (hasRangeBackedPerLayerInputs && !canUseHotVocabularyBatchDecode) {
    return generateNTokensGPUStepwiseRangeBackedPle(
      state,
      N,
      currentIds,
      opts,
      helpers
    );
  }
  if (canUseHotVocabularyBatchDecode) {
    useGpuStopFlags = true;
    effectiveStopCheckMode = 'per-token';
  }

  const stopTokenIds = config.stopTokenIds;
  const eosToken = state.tokenizer?.getSpecialTokens?.()?.eos;
  const padTokenId = state.tokenizer?.getSpecialTokens?.()?.pad ?? null;
  const logitSoftcap = config.finalLogitSoftcapping === null
    ? 0
    : config.finalLogitSoftcapping;
  if (eosToken == null && stopTokenIds.length === 0) {
    throw new Error('[Pipeline] Missing EOS token. Ensure tokenizer or manifest provides stop tokens.');
  }
  const eosTokenId = eosToken ?? stopTokenIds[0];
  if (eosTokenId == null) {
    throw new Error('[Pipeline] Missing EOS token. Ensure tokenizer or manifest provides stop tokens.');
  }
  const maxTokens = executionPlan?.maxTokens
    ?? opts.maxTokens
    ?? state.runtimeConfig.inference.generation.maxTokens;
  const maxSeqLen = state.currentSeqLen + maxTokens;

  const recordStart = performance.now();

  const ring = state.decodeRing;
  let ringSlot = null;
  if (ring) {
    ring.ensure({
      batchSize,
      tokensPerInterval,
      stopCheckMode: effectiveStopCheckMode,
      ringTokens: executionPlan?.ringTokens ?? state.runtimeConfig.inference.batching.ringTokens,
      ringStop: executionPlan?.ringStop ?? state.runtimeConfig.inference.batching.ringStop,
      ringStaging: executionPlan?.ringStaging ?? state.runtimeConfig.inference.batching.ringStaging,
    });
    ringSlot = ring.acquire();
  }

  const tokenCapacity = ringSlot?.tokens ? ringSlot.tokensPerInterval : N;
  const tokensBuffer = ringSlot?.tokens ?? device.createBuffer({
    size: (tokenCapacity + 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const ownsTokensBuffer = !ringSlot?.tokens;

  const stopCapacity = ringSlot?.stop ? ringSlot.tokensPerInterval + 1 : N + 1;
  const stopBuffer = useGpuStopFlags
    ? ringSlot?.stop ?? device.createBuffer({
      size: stopCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    : null;
  const ownsStopBuffer = useGpuStopFlags && !ringSlot?.stop;
  const pleInputTokensBuffer = canUseHotVocabularyBatchDecode
    ? device.createBuffer({
      size: (tokenCapacity + 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'ple_hot_input_tokens',
    })
    : null;

  const tokensStagingBuffer = ringSlot?.stagingTokens ?? device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const ownsTokensStaging = !ringSlot?.stagingTokens;

  const stopStagingBuffer = useGpuStopFlags
    ? ringSlot?.stagingStop ?? device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    : null;
  const ownsStopStaging = useGpuStopFlags && !ringSlot?.stagingStop;
  const finitenessStagingBuffer = state.finitenessBuffer
    ? ringSlot?.stagingFiniteness ?? device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    : null;
  const ownsFinitenessStaging = Boolean(state.finitenessBuffer) && !ringSlot?.stagingFiniteness;
  let readbackCleanupDelegated = false;
  let repHistoryBuffer = null;
  let repHistoryCount = 0;
  try {
    if (state.finitenessBuffer) {
      device.queue.writeBuffer(state.finitenessBuffer, 0, new Uint32Array([0, 0, 0, 0]));
    }

    device.queue.writeBuffer(tokensBuffer, 0, new Uint32Array([startToken]));
    if (pleInputTokensBuffer) {
      device.queue.writeBuffer(pleInputTokensBuffer, 0, new Uint32Array([hotStartTokenIndex]));
    }
    if (stopBuffer) {
      const stopElements = stopBuffer.size / 4;
      const zeroStopData = ringSlot?.zeroStopData;
      const clearData = zeroStopData && zeroStopData.length <= stopElements
        ? zeroStopData
        : new Uint32Array(stopElements);
      device.queue.writeBuffer(stopBuffer, 0, clearData);
    }

    const context = helpers.buildLayerContext(recorder, true, opts.debugLayers, executionPlan);
    const embedBufferRaw = state.weights.get('embed');
    if (isCpuWeightBuffer(embedBufferRaw)) {
      throw new Error('[Pipeline] GPU-only decode not supported with CPU-resident embeddings.');
    }
    if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw)) {
      throw new Error('Embed buffer not found or not a GPUBuffer/WeightBuffer');
    }
    const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
    const embedDtype = getWeightDtype(embedBufferRaw);
    const activationDtype = getEffectiveActivationDtype(state, opts);

    // GPU-side repetition penalty: upload deduplicated history before batch
    const repetitionPenalty = opts.repetitionPenalty ?? samplingDefaults.repetitionPenalty;
    const repPenaltyWindow = samplingDefaults.repetitionPenaltyWindow;
    if (repetitionPenalty !== 1.0 && currentIds.length > 0) {
      const uniqueTokens = [...new Set(currentIds.slice(-repPenaltyWindow))];
      repHistoryCount = uniqueTokens.length;
      const historyData = new Uint32Array(uniqueTokens);
      repHistoryBuffer = device.createBuffer({
        size: Math.max(4, historyData.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'rep_penalty_history',
      });
      device.queue.writeBuffer(repHistoryBuffer, 0, historyData);
    }

    // Hoist loop-invariant values to avoid repeated rule lookups and allocations.
    const embeddingDtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype: embedDtype });
    const debugProbes = state.runtimeConfig.shared.debug.probes;
    const currentTokenIdsArray = [startToken];

    for (let i = 0; i < N; i++) {
      currentTokenIdsArray[0] = rollingIds[i];
      const currentPos = state.currentSeqLen + i;
      context.currentSeqLen = currentPos;
      context.currentTokenIds = currentTokenIdsArray;
      context.decodeBuffers?.resetPingPong();

      const hiddenTensor = await embed(tokensBuffer, embedBuffer, {
        hiddenSize: config.hiddenSize,
        vocabSize: config.vocabSize,
        scaleEmbeddings: config.scaleEmbeddings,
        recorder,
        transpose: state.embeddingTranspose,
        debugProbes,
        operatorDiagnostics: state.operatorDiagnostics,
        activationDtype,
        embeddingDtype,
        executionPolicies: state.executionV1State?.policies ?? null,
        numTokens: 1,
        indexOffset: i,
      });

      let hiddenStatesBuffer = hiddenTensor.buffer;
      const perLayerInputs = await preparePerLayerInputs(tokensBuffer, hiddenTensor, context, {
        numTokens: 1,
        indexOffset: i,
        perLayerTokenIds: pleInputTokensBuffer,
        perLayerIndexOffset: i,
        tokenIdHint: Number.isInteger(rollingIds[i]) ? rollingIds[i] : null,
        pleCache: state.pleCache ?? null,
      });
      try {
        for (let l = 0; l < config.numLayers; l++) {
          context.perLayerInputBuffer = perLayerInputs?.[l] ?? null;
          const prevStates = hiddenStatesBuffer;
          hiddenStatesBuffer = (await processLayer(l, hiddenStatesBuffer, 1, false, context));
          context.decodeBuffers?.swapPingPong();
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
          if (isGpuBufferInstance(prevStates) && prevStates !== hiddenStatesBuffer) {
            const ownsBuffer = context.decodeBuffers?.ownsBuffer(prevStates);
            if (!ownsBuffer) {
              recorder.trackTemporaryBuffer(prevStates);
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

      const logits = await recordLogitsGPU(
        recorder,
        hiddenStatesBuffer,
        1,
        helpers.getLogitsWeights(),
        helpers.getLogitsConfig(),
        state.operatorDiagnostics
      );
      const { logitsBuffer, vocabSize, logitsDtype } = logits;

      // Apply GPU-side repetition penalty before sampling
      if (repHistoryBuffer && repetitionPenalty !== 1.0) {
        await recordRepPenalty(recorder, logitsBuffer, repHistoryBuffer, tokensBuffer, {
          vocabSize,
          historyCount: repHistoryCount,
          penalty: repetitionPenalty,
          batchCount: i,
          batchOffset: 1,
          logitsDtype,
        });
      }

      const outputIndex = i + 1;
      if (opts.temperature < samplingDefaults.greedyThreshold) {
        await recordArgmax(recorder, logitsBuffer, vocabSize, {
          padTokenId,
          logitSoftcap,
          logitsDtype,
          outputBuffer: tokensBuffer,
          outputIndex,
        });
      } else {
        await recordGPUSample(recorder, logitsBuffer, vocabSize, {
          temperature: opts.temperature,
          topK: opts.topK,
          padTokenId,
          logitSoftcap,
          logitsDtype,
          outputBuffer: tokensBuffer,
          outputIndex,
          greedyThreshold: samplingDefaults.greedyThreshold,
        });
      }

      const stopCheck = canUseHotVocabularyBatchDecode
        ? recordCheckHotVocabStop(recorder, {
          sampledTokenBuffer: tokensBuffer,
          nextInputTokenBuffer: pleInputTokensBuffer,
          hotTokenIndexMapBuffer: pleHotVocabularyRuntime.hotTokenIndexMapBuffer,
          hotTokenSentinel: pleHotVocabularyRuntime.sentinelIndex,
          shouldStopBuffer: stopBuffer,
          tokenIndex: outputIndex,
          eosTokenId,
          maxTokens: maxSeqLen,
          currentPos,
        })
        : useGpuStopFlags
        ? recordCheckStop(recorder, {
          sampledTokenBuffer: tokensBuffer,
          shouldStopBuffer: stopBuffer,
          tokenIndex: outputIndex,
          eosTokenId,
          maxTokens: maxSeqLen,
          currentPos,
        })
        : null;

      if (isGpuBufferInstance(hiddenStatesBuffer) && !context.decodeBuffers?.ownsBuffer(hiddenStatesBuffer)) {
        recorder.trackTemporaryBuffer(hiddenStatesBuffer);
      }
      if (isGpuBufferInstance(logitsBuffer)) {
        recorder.trackTemporaryBuffer(logitsBuffer);
      }
      if (isGpuBufferInstance(stopCheck) && stopCheck !== stopBuffer) {
        recorder.trackTemporaryBuffer(stopCheck);
      }
    }

    const recordMs = performance.now() - recordStart;
    state.stats.decodeRecordMs = (state.stats.decodeRecordMs ?? 0) + recordMs;

    const encoder = recorder.getEncoder();
    encoder.copyBufferToBuffer(tokensBuffer, 4, tokensStagingBuffer, 0, N * 4);
    if (useGpuStopFlags && stopBuffer && stopStagingBuffer) {
      encoder.copyBufferToBuffer(stopBuffer, 4, stopStagingBuffer, 0, N * 4);
    }

    if (state.finitenessBuffer && finitenessStagingBuffer) {
      encoder.copyBufferToBuffer(state.finitenessBuffer, 0, finitenessStagingBuffer, 0, 16);
    }

    if (!allowReadback('pipeline.decode.sample')) {
      throw new Error('[Pipeline] GPU readback disabled for sampling');
    }

    recorder.submit({ cleanup: 'deferred' });

    const readbackStart = performance.now();
    readbackCleanupDelegated = true;
    const readback = await readBatchTokensFromStagingBuffers({
      tokensStagingBuffer,
      stopStagingBuffer,
      finitenessStagingBuffer,
      tokenCount: N,
      ownsTokensStaging,
      ownsStopStaging,
      ownsFinitenessStaging,
      ring,
      cleanupRecorder: recorder,
    });
    const readbackWaitMs = performance.now() - readbackStart;
    state.stats.decodeReadbackWaitMs = (state.stats.decodeReadbackWaitMs ?? 0) + readbackWaitMs;

    const isInfinite = readback.finitenessStatus.triggered;
    const metadata = readback.finitenessStatus.metadata;

    const submitWaitMs = recorder.getSubmitLatencyMs();
    if (submitWaitMs != null) {
      state.stats.decodeSubmitWaitMs = (state.stats.decodeSubmitWaitMs ?? 0) + submitWaitMs;
    }

    getUniformCache().flushPendingDestruction();

    const tokens = readback.tokens;
    const stopFlags = readback.stopFlags;

    if (stopFlags) {
      log.debug('Pipeline', `[STOP] N=${N} flags=[${Array.from(stopFlags).join(',')}] tokens=[${tokens.join(',')}] eos=${eosTokenId}`);
    }

    const actualCount = resolveBatchStop(tokens, stopFlags, stopTokenIds, eosToken);
    const generatedTokens = tokens.slice(0, actualCount);
    const invalidToken = findInvalidGeneratedToken(generatedTokens, config.vocabSize, padTokenId);

    if (isInfinite) {
      throw new FinitenessError(`F16 bounds exceeded during batch generation${metadata}`);
    }
    if (invalidToken) {
      state.disableFusedDecode = true;
      throw new Error(
        `[Pipeline] Batch decode produced invalid token ${invalidToken.tokenId} ` +
        `at batch index ${invalidToken.index} (vocabSize=${config.vocabSize}, padTokenId=${padTokenId ?? 'none'}).`
      );
    }

    if (opts.profile && recorder.isProfilingEnabled()) {
      const timings = await recorder.resolveProfileTimings();
      const total = sumProfileTimings(timings);
      if (total !== null) {
        state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
      }
      if (timings) {
        recordDecodeProfileStep(state, {
          batch: true,
          stepStart: state.decodeStepCount + 1,
          stepCount: actualCount,
          timings,
          totalMs: total ?? undefined,
        });
        const stepStart = state.decodeStepCount + 1;
        if (shouldLogProfileStep(state, stepStart)) {
          log.warn('Profile', `Batch decode (N=${N}):`);
          log.warn('Profile', CommandRecorder.formatProfileReport(timings));
        }
      }
    }

    state.currentSeqLen += actualCount;
    return { tokens: generatedTokens, actualCount };
  } finally {
    state.batchingStats.totalBatchedTimeMs += Math.max(0, performance.now() - batchStart);
    state.batchingStats.gpuSubmissions += 1;

    if (!readbackCleanupDelegated) {
      if (ownsFinitenessStaging) {
        finitenessStagingBuffer.destroy();
      }
      if (ownsTokensStaging) tokensStagingBuffer.destroy();
      if (ownsStopStaging) stopStagingBuffer?.destroy();
      ring?.advance();
    }
    if (ownsTokensBuffer) tokensBuffer.destroy();
    if (ownsStopBuffer) stopBuffer?.destroy();
    pleInputTokensBuffer?.destroy();
    if (repHistoryBuffer) repHistoryBuffer.destroy();
  }
}
