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
import { resolvePerLayerInputsSession } from './generator/session-context.js';
import { processLayer } from './layer.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, recordGreedyLmHeadArgmaxGPU, extractLastPositionLogits } from './logits/index.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer, getWeightDtype, getWeightMetadata } from '../../../gpu/weight-buffer.js';
import { decodeReadback } from './debug-utils/index.js';
import { captureObservedFusedDecodeLogits, emitObservedLogits } from './generator-logits-observation.js';
import { getFinalNormWeights, extractEmbeddingFromHidden } from './generator-runtime.js';
import { parseFinitenessStatusWords } from './finiteness-guard-status.js';
import { hasLinearAttentionLayers } from './linear-attention.js';
import { hasConvLayers } from './layer.js';
import { advanceDecodeStepCount } from './tsir-fixture-writer.js';
import {
  preparePerLayerInputs,
  createPleBufferCache,
  prefetchPerLayerRow,
  hasRangeBackedPerLayerInputEmbeddings,
  hasGpuSplitPerLayerInputEmbeddings,
  getPleHotVocabularyRuntime,
} from './per-layer-inputs.js';
import { FINITENESS_RESET_WORDS, FinitenessError, decodeStep, getEffectiveActivationDtype, isOwnedDecodeBuffer, recordDecodeProfileStep, releasePerLayerInputBuffer, runDecodeLayers, shouldLogProfileStep, sumProfileTimings } from './generator/decode.js';
export { FinitenessError, decodeStep, decodeStepLogits, readMappedBufferCopy, readSampledTokenFromStagingBuffer, shouldUseFusedDecodeSampling, sumProfileTimings } from './generator/decode.js';

const FINITENESS_STATUS_BYTES = FINITENESS_RESET_WORDS.byteLength;

function mergeRecordOpLabelCounts(target, source) {
  const merged = target && typeof target === 'object' && !Array.isArray(target)
    ? target
    : {};
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    return merged;
  }
  for (const [label, rawCount] of Object.entries(source)) {
    const count = Number(rawCount);
    if (typeof label !== 'string' || label.length === 0 || !Number.isFinite(count) || count <= 0) {
      continue;
    }
    merged[label] = (Number.isFinite(merged[label]) ? merged[label] : 0) + count;
  }
  return merged;
}

export function shouldUseBatchDecode(config) {
  return isBatchDecodeEnabled(config);
}

function shouldUseGreedyLmHeadArgmaxFusion(state, opts, samplingDefaults, repetitionPenalty) {
  const probes = state.runtimeConfig?.shared?.debug?.probes;
  return state.runtimeConfig?.inference?.session?.useGreedyLmHeadArgmaxFusion === true && (state.modelConfig?.logitOutputScale ?? 1) === 1
    && opts.temperature < samplingDefaults.greedyThreshold
    && repetitionPenalty === 1.0
    && state.operatorDiagnostics == null
    && !(Array.isArray(probes) && probes.length > 0);
}

export function createStopTokenLookup(stopTokenIds, eosTokenId) {
  if (!Array.isArray(stopTokenIds)) {
    throw new Error('[Pipeline] stopTokenIds must be an array.');
  }

  const tokenIds = [];
  for (const tokenId of stopTokenIds) {
    if (!tokenIds.includes(tokenId)) {
      tokenIds.push(tokenId);
    }
  }
  if (typeof eosTokenId === 'number' && !tokenIds.includes(eosTokenId)) {
    tokenIds.push(eosTokenId);
  }

  if (tokenIds.length === 0) {
    return { firstTokenId: null, secondTokenId: null, tokenSet: null };
  }
  if (tokenIds.length === 1) {
    return { firstTokenId: tokenIds[0], secondTokenId: null, tokenSet: null };
  }
  if (tokenIds.length === 2) {
    return { firstTokenId: tokenIds[0], secondTokenId: tokenIds[1], tokenSet: null };
  }
  return { firstTokenId: null, secondTokenId: null, tokenSet: new Set(tokenIds) };
}

function lookupHasStopToken(token, stopTokenLookup) {
  if (stopTokenLookup.tokenSet) {
    return stopTokenLookup.tokenSet.has(token);
  }
  if (stopTokenLookup.firstTokenId !== null && token === stopTokenLookup.firstTokenId) {
    return true;
  }
  return stopTokenLookup.secondTokenId !== null && token === stopTokenLookup.secondTokenId;
}

export function resolveBatchStop(tokens, stopFlags, stopTokenLookup) {
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
    if (lookupHasStopToken(tokens[i], stopTokenLookup)) {
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

export async function readBatchTokensFromStagingBuffers(options) {
  const {
    tokensStagingBuffer,
    stopStagingBuffer = null,
    finitenessStagingBuffer = null,
    finitenessOffsetBytes = null,
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
  const hasPackedFiniteness = Number.isFinite(finitenessOffsetBytes) && finitenessOffsetBytes >= 0;
  const timing = {
    mapWaitMs: 0,
    cleanupMs: 0,
    copyMs: 0,
  };

  try {
    const mapStart = performance.now();
    const mapPromises = [tokensStagingBuffer.mapAsync(GPUMapMode.READ)];
    if (stopStagingBuffer) {
      mapPromises.push(stopStagingBuffer.mapAsync(GPUMapMode.READ));
    }
    if (finitenessStagingBuffer && !hasPackedFiniteness) {
      mapPromises.push(finitenessStagingBuffer.mapAsync(GPUMapMode.READ));
    }
    const mapResults = await Promise.allSettled(mapPromises);
    timing.mapWaitMs = performance.now() - mapStart;
    tokensMapped = mapResults[0]?.status === 'fulfilled';
    stopMapped = Boolean(stopStagingBuffer) && mapResults[1]?.status === 'fulfilled';
    finitenessMapped = Boolean(finitenessStagingBuffer) && !hasPackedFiniteness
      && mapResults[stopStagingBuffer ? 2 : 1]?.status === 'fulfilled';
    const mapFailure = mapResults.find((result) => result.status === 'rejected');
    if (mapFailure) {
      throw mapFailure.reason;
    }
    const cleanupStart = performance.now();
    await cleanupRecorder?.completeDeferredCleanup();
    timing.cleanupMs = performance.now() - cleanupStart;
    cleanupCompleted = true;

    const copyStart = performance.now();
    const tokensRange = tokensStagingBuffer.getMappedRange();
    const tokenWords = new Uint32Array(tokensRange, 0, tokenCount);
    const tokens = new Uint32Array(tokenWords.length);
    tokens.set(tokenWords);
    let stopFlags = null;
    if (stopStagingBuffer) {
      const stopWords = new Uint32Array(stopStagingBuffer.getMappedRange()).subarray(0, tokenCount);
      stopFlags = new Uint32Array(stopWords.length);
      stopFlags.set(stopWords);
    }
    const finitenessStatus = hasPackedFiniteness
      ? parseFinitenessStatusWords(new Uint32Array(tokensRange, finitenessOffsetBytes, 4), 0)
      : finitenessStagingBuffer
      ? parseFinitenessStatusWords(new Uint32Array(finitenessStagingBuffer.getMappedRange()), 0)
      : { triggered: false, metadata: '' };
    timing.copyMs = performance.now() - copyStart;

    return {
      tokens,
      stopFlags,
      finitenessStatus,
      timing,
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

export async function advanceWithToken(state, tokenId, opts, helpers) {
  advanceDecodeStepCount(state);

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

  advanceDecodeStepCount(state);

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
      state.embeddingPostprocessor,
      null,
      state.weights.get('final_norm_bias') ?? null
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
  const resolvedPleSession = resolvePerLayerInputsSession(
    config.perLayerInputsSession ?? null,
    state.runtimeConfig?.inference?.session?.perLayerInputs ?? null
  );
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
        resolvedPleSession
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
  const recorderOptions = {
    recordLabels: opts.debug === true || opts.benchmark === true || opts.executionObserver === true,
    recordDispatches: opts.debug === true || opts.executionObserver === true,
    aggregateDispatches: opts.executionObserver === true,
  };
  const recorder = opts.profile
    ? createProfilingRecorder('batch_decode', device, recorderOptions)
    : createCommandRecorder('batch_decode', recorderOptions, device);
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
  const hasGpuSplitPerLayerInputs = hasGpuSplitPerLayerInputEmbeddings({
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
  const stopTokenLookup = createStopTokenLookup(stopTokenIds, eosToken);
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

  const tokenReadbackBytes = N * 4;
  const tokenStagingBytes = tokenReadbackBytes + (state.finitenessBuffer ? FINITENESS_STATUS_BYTES : 0);
  const tokensStagingBuffer = ringSlot?.stagingTokens ?? device.createBuffer({
    size: tokenStagingBytes,
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
  const packedFinitenessOffsetBytes = state.finitenessBuffer
    && tokensStagingBuffer.size >= tokenReadbackBytes + FINITENESS_STATUS_BYTES
      ? tokenReadbackBytes
      : null;
  const finitenessStagingBuffer = state.finitenessBuffer && packedFinitenessOffsetBytes == null
    ? ringSlot?.stagingFiniteness ?? device.createBuffer({
      size: FINITENESS_STATUS_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    : null;
  const ownsFinitenessStaging = Boolean(state.finitenessBuffer)
    && packedFinitenessOffsetBytes == null
    && !ringSlot?.stagingFiniteness;
  let readbackCleanupDelegated = false;
  let repHistoryBuffer = null;
  let repHistoryCount = 0;
  try {
    if (state.finitenessBuffer) {
      device.queue.writeBuffer(state.finitenessBuffer, 0, FINITENESS_RESET_WORDS);
    }

    const singleWordUpload = new Uint32Array(1);
    singleWordUpload[0] = startToken;
    device.queue.writeBuffer(tokensBuffer, 0, singleWordUpload);
    if (pleInputTokensBuffer) {
      singleWordUpload[0] = hotStartTokenIndex;
      device.queue.writeBuffer(pleInputTokensBuffer, 0, singleWordUpload);
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
    if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw) && !isSplitWeightBuffer(embedBufferRaw)) {
      throw new Error('Embed buffer not found or not a GPUBuffer/WeightBuffer');
    }
    const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
    const embedDtype = getWeightDtype(embedBufferRaw);
    const embedMetadata = getWeightMetadata(embedBufferRaw);
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
    const embeddingDtype = selectRuleValue('inference', 'dtype', 'embeddingDtype', { dtype: embedDtype });
    const debugProbes = state.runtimeConfig.shared.debug.probes;
    const currentTokenIdsArray = [startToken];
    const useGreedyLmHeadArgmaxFusion = shouldUseGreedyLmHeadArgmaxFusion(
      state,
      opts,
      samplingDefaults,
      repetitionPenalty
    );

    for (let i = 0; i < N; i++) {
      // In the GPU batch path, only the start token (i=0) is known on the CPU.
      // Subsequent tokens (i>0) are sampled on the GPU and not read back until
      // after the full batch completes.  Set currentTokenIds to null for those
      // iterations so downstream code (PLE cache, KV-cache update) gracefully
      // skips CPU-side token-dependent optimizations.
      currentTokenIdsArray[0] = i === 0 ? startToken : null;
      const currentPos = state.currentSeqLen + i;
      context.currentSeqLen = currentPos;
      context.currentTokenIds = currentTokenIdsArray;
      context.decodeBuffers?.resetPingPong();

      const hiddenTensor = await embed(tokensBuffer, embedBuffer, {
        hiddenSize: config.hiddenSize,
        vocabSize: config.vocabSize,
        scaleEmbeddings: config.scaleEmbeddings,
        embeddingScale: config.embeddingScale, embeddingNormalization: config.embeddingNormalization,
        recorder,
        outputBuffer: context.decodeBuffers?.getHiddenBuffer() ?? undefined,
        transpose: state.embeddingTranspose,
        debugProbes,
        operatorDiagnostics: state.operatorDiagnostics,
        activationDtype,
        embeddingDtype,
        embeddingStorageEncoding: embedMetadata?.storageEncoding ?? null,
        executionPolicies: state.executionV1State?.policies ?? null,
        numTokens: 1,
        indexOffset: i,
      });

      let hiddenStatesBuffer = hiddenTensor.buffer;
      const perLayerInputs = await preparePerLayerInputs(tokensBuffer, hiddenTensor, context, {
        numTokens: 1,
        indexOffset: i,
        perLayerTokenIds: pleInputTokensBuffer ?? (hasGpuSplitPerLayerInputs ? tokensBuffer : null),
        perLayerIndexOffset: i,
        tokenIdHint: i === 0 ? startToken : null,
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

      const outputIndex = i + 1;
      let logitsBuffer = null;
      if (useGreedyLmHeadArgmaxFusion) {
        await recordGreedyLmHeadArgmaxGPU(
          recorder,
          hiddenStatesBuffer,
          1,
          helpers.getLogitsWeights(),
          helpers.getLogitsConfig(),
          {
            padTokenId,
            logitSoftcap,
            outputBuffer: tokensBuffer,
            outputIndex,
          },
          state.operatorDiagnostics
        );
      } else {
        const logits = await recordLogitsGPU(
          recorder,
          hiddenStatesBuffer,
          1,
          helpers.getLogitsWeights(),
          helpers.getLogitsConfig(),
          state.operatorDiagnostics
        );
        const { vocabSize, logitsDtype } = logits;
        logitsBuffer = logits.logitsBuffer;

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
      }

      const stopCheck = canUseHotVocabularyBatchDecode
        ? await recordCheckHotVocabStop(recorder, {
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
        ? await recordCheckStop(recorder, {
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
    const recordStats = recorder.getStats();
    const recordOps = recordStats.opCount;
    const recordPasses = recordStats.computePassCount;
    state.stats.decodeRecordMs = (state.stats.decodeRecordMs ?? 0) + recordMs;
    state.stats.decodeRecordOps = (state.stats.decodeRecordOps ?? 0) + recordOps;
    state.stats.decodeRecordPasses = (state.stats.decodeRecordPasses ?? 0) + recordPasses;
    state.stats.decodeRecordOpLabels = mergeRecordOpLabelCounts(
      state.stats.decodeRecordOpLabels,
      recordStats.opLabelCounts
    );

    const encoder = recorder.getEncoder();
    encoder.copyBufferToBuffer(tokensBuffer, 4, tokensStagingBuffer, 0, N * 4);
    if (useGpuStopFlags && stopBuffer && stopStagingBuffer) {
      encoder.copyBufferToBuffer(stopBuffer, 4, stopStagingBuffer, 0, N * 4);
    }

    if (state.finitenessBuffer && packedFinitenessOffsetBytes != null) {
      encoder.copyBufferToBuffer(
        state.finitenessBuffer,
        0,
        tokensStagingBuffer,
        packedFinitenessOffsetBytes,
        FINITENESS_STATUS_BYTES
      );
    } else if (state.finitenessBuffer && finitenessStagingBuffer) {
      encoder.copyBufferToBuffer(
        state.finitenessBuffer,
        0,
        finitenessStagingBuffer,
        0,
        FINITENESS_STATUS_BYTES
      );
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
      finitenessOffsetBytes: packedFinitenessOffsetBytes,
      tokenCount: N,
      ownsTokensStaging,
      ownsStopStaging,
      ownsFinitenessStaging,
      ring,
      cleanupRecorder: recorder,
    });
    const readbackWaitMs = performance.now() - readbackStart;
    state.stats.decodeReadbackWaitMs = (state.stats.decodeReadbackWaitMs ?? 0) + readbackWaitMs;
    if (readback.timing) {
      state.stats.decodeReadbackMapWaitMs = (state.stats.decodeReadbackMapWaitMs ?? 0)
        + readback.timing.mapWaitMs;
      state.stats.decodeReadbackCleanupMs = (state.stats.decodeReadbackCleanupMs ?? 0)
        + readback.timing.cleanupMs;
      state.stats.decodeReadbackCopyMs = (state.stats.decodeReadbackCopyMs ?? 0)
        + readback.timing.copyMs;
    }

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

    const actualCount = resolveBatchStop(tokens, stopFlags, stopTokenLookup);
    const generatedTokens = actualCount === tokens.length
      ? tokens
      : tokens.subarray(0, actualCount);
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
    state.batchingStats.executedBatchTokens = (state.batchingStats.executedBatchTokens ?? 0) + N;
    state.batchingStats.resolvedBatchTokens = (state.batchingStats.resolvedBatchTokens ?? 0) + actualCount;

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
          batch: true,
          stepStart: state.decodeStepCount + 1,
          stepCount: actualCount,
          timings,
          totalMs: total ?? undefined,
          recorderStats: recorder.getStats(),
        });
        const stepStart = state.decodeStepCount + 1;
        if (!opts.executionObserver && shouldLogProfileStep(state, stepStart)) {
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
