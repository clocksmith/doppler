import { getDevice, setTrackSubmits } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer, readBufferSlice, uploadData } from '../../../../memory/buffer-pool.js';
import { resetSubmitStats, logSubmitStats } from '../../../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../../../gpu/command-recorder.js';
import { allowReadback } from '../../../../gpu/perf-guards.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from '../sampling.js';
import { embed } from '../embed.js';
import { runProbes } from '../probes.js';
import { processLayer } from '../layer.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, extractLastPositionLogits } from '../logits/index.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer, getWeightDtype, getWeightMetadata, getLayout } from '../../../../gpu/weight-buffer.js';
import {
  decodeStep,
  decodeStepLogits,
  advanceWithToken,
  generateNTokensGPU,
  shouldUseBatchDecode,
  sumProfileTimings,
  FinitenessError,
  advanceWithTokenAndEmbedding as runAdvanceWithTokenAndEmbedding,
} from '../generator-steps.js';
import {
  buildLayerContext,
  debugCheckBuffer as debugCheckBufferHelper,
  resolvePerLayerInputsSession,
} from './session-context.js';
import {
  getLogitsConfig,
  getLogitsWeights,
} from './logits-config.js';
import { releaseSharedAttentionState } from './attention-lifecycle.js';
import { decodeReadback, getLogitsHealth } from '../debug-utils/index.js';
import { parseFinitenessStatusWords } from '../finiteness-guard-status.js';
import {
  advanceDecodeStepCount,
  createTsirFixtureState,
  drainPendingTsirReads,
} from '../tsir-fixture-writer.js';
import {
  activateFallbackExecutionPlan,
  hasFallbackExecutionPlan,
  rebaseExecutionSessionPlan,
  resetActiveExecutionPlan,
  resolveMaxBatchDecodeTokens,
  resolvePrefillRecorderChunkLayers,
  resolveActiveExecutionPlan,
  setActiveExecutionPlan,
} from '../execution-plan.js';
import {
  cloneLinearAttentionRuntime,
  hasLinearAttentionLayers,
  resetLinearAttentionRuntime,
  restoreLinearAttentionRuntime,
} from '../linear-attention.js';
import {
  preparePerLayerInputs,
  createPleBufferCache,
  prefetchPerLayerRow,
  getPleHotVocabularyRuntime,
  hasGpuSplitPerLayerInputEmbeddings,
  hasRangeBackedPerLayerInputEmbeddings,
} from '../per-layer-inputs.js';
import { createTensor } from '../../../../gpu/tensor.js';
import {
  releasePerLayerInputBuffer,
  shouldDisablePrefillCommandBatching,
  resolveEffectivePrefillTokenChunkSize,
} from './prefill-policy.js';
import {
  normalizePrefixEmbeddingOverride,
  resolvePrefillEmbeddingInputIds,
  resolvePrefillMultimodalBidirectionalSpan,
  applyPrefixEmbeddingOverride,
  resolvePrefixEmbeddingOverrideTransitionDeclaredBy,
} from './prefix-embedding.js';
import {
  shouldDisableBatchDecodeAfterShortBatch,
  resolveHotVocabularyBatchDecodeAvailability,
  primePleDecodeRuntimeCache,
  recordPrefillProfileStep,
  resolveTokenText,
  usesReplayPrefillDecode,
  assertIncrementalDecodeSupport,
  summarizeExecutionPlan,
  shouldRetryWithFinitenessFallback,
  createUnhandledFinitenessPolicyError,
  resolveTargetPlanKVDtype,
  resolveCurrentKVCacheDtype,
  cloneRuntimeInferenceWithKVDtype,
} from '../generator-decode-policy.js';
import { FINITENESS_RESET_WORDS, borrowLinearWeight, borrowNormWeight, canUseChunkedSoftEmbeddingLogits, getExperimentalIntentBundleModule, normalizeCanvasTokenIds, normalizeSelectedLogitTokenIds, normalizeSelfConditioningLogits, normalizeSelfConditioningLogitsState, normalizeSelfConditioningSoftEmbeddingState, recordPrefillRecorderStats, releaseBorrowedWeight, resolveDiffusionGemmaSoftEmbeddingChunkRows, resolvePrefillChunkSubmitMode, resolveSuppressedSamplingTokenIds, traceActivationHealth } from './text.js';

export async function _commitPrefillHiddenChunk(prefillResult) {
    const {
      numTokens,
      startPos,
      currentRecorder,
      recordProfile,
      currentHiddenBuffer,
    } = prefillResult;

    try {
      if (currentRecorder) {
        recordPrefillRecorderStats(this._state, currentRecorder);
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
      } else {
        const device = getDevice();
        if (device) {
          await device.queue.onSubmittedWorkDone();
        }
      }
      this._state.currentSeqLen = startPos + numTokens;
    } finally {
      releaseBuffer(currentHiddenBuffer);
    }
  }

export async function _prefillInputIdsToLogits(inputIds, opts) {
    const chunkSize = this._resolvePrefillTokenChunkSize(inputIds);
    if (chunkSize === null) {
      return this._prefill(inputIds, opts);
    }

    for (let offset = 0; offset < inputIds.length; offset += chunkSize) {
      const end = Math.min(offset + chunkSize, inputIds.length);
      const chunk = inputIds.slice(offset, end);
      const isFinalChunk = end === inputIds.length;
      if (isFinalChunk) {
        return this._prefill(chunk, opts);
      }
      const prefillResult = await this._prefillToHidden(chunk, opts);
      await this._commitPrefillHiddenChunk(prefillResult);
    }

    return this._prefill(inputIds, opts);
  }

export async function _prefillPromptToLogits(prompt, opts, contextLabel) {
    const prefillStartSeqLen = this._state.currentSeqLen;
    const inputStart = performance.now();
    const inputIds = this._resolvePromptOrInputIds(prompt, opts.useChatTemplate, contextLabel, opts.inputIds);
    const inputMs = performance.now() - inputStart;
    if (opts.debug) {
      log.debug('Pipeline', `${contextLabel}: ${inputIds.length} tokens`);
    }

    let logits;
    const prefillStart = performance.now();
    const runPrefill = () => this._prefillInputIdsToLogits(inputIds, opts);
    try {
      logits = await runPrefill();
    } catch (error) {
      if (!this._shouldUseFinitenessFallback(error, contextLabel)) {
        throw error;
      }
      log.warn('Pipeline', `FinitenessGuard caught NaN/Inf during ${contextLabel}. Retrying with F32 precision.`);
      logits = await this._retryWithPersistentFinitenessFallback(
        opts,
        contextLabel,
        opts.maxTokens ?? 1,
        runPrefill,
        prefillStartSeqLen
      );
    }

    return {
      inputIds,
      logits,
      phase: {
        inputMs,
        prefillMs: performance.now() - prefillStart,
        tokens: inputIds.length,
        prefillRecordMs: this._state.stats.prefillRecordMs ?? null,
        prefillRecordOps: this._state.stats.prefillRecordOps ?? null,
        prefillRecordPasses: this._state.stats.prefillRecordPasses ?? null,
        prefillRecordOpLabels: this._state.stats.prefillRecordOpLabels ?? null,
        prefillSubmitWaitMs: this._state.stats.prefillSubmitWaitMs ?? null,
        gpuPrefillMs: this._state.stats.gpuTimePrefillMs ?? null,
      },
    };
  }

export async function _prefillToHidden(inputIds, opts) {
    // Internal-only: reuse the main prefill implementation but stop before logits.
    return this._prefill(inputIds, { ...opts, _returnHidden: true });
  }

export async function _prefill(inputIds, opts) {
    const numTokens = inputIds.length;
    const config = this._state.modelConfig;
    const selectedLogitTokenIds = normalizeSelectedLogitTokenIds(
      opts?._selectedLogitTokenIds,
      config.vocabSize,
      '_prefill.selectedLogitTokenIds'
    );
    const startPos = this._state.currentSeqLen;
    const tracePrefillEnabled = isTraceEnabled('perf');
    const prefillTraceStart = tracePrefillEnabled ? performance.now() : 0;
    const returnHidden = opts?._returnHidden === true;
    const embeddingInputIds = resolvePrefillEmbeddingInputIds(
      inputIds,
      opts?.embeddingInputSpan ?? null,
      '_prefill'
    );
    const multimodalBidirectionalSpan = resolvePrefillMultimodalBidirectionalSpan(
      inputIds,
      opts?.multimodalBidirectionalSpan ?? null,
      '_prefill'
    );
    if (embeddingInputIds !== inputIds) {
      this._assertTokenIdsInRange(embeddingInputIds, '_prefill.embeddingInputIds');
    }
    const embeddingOverride = normalizePrefixEmbeddingOverride(
      opts?.embeddingOverrides ?? null,
      config.hiddenSize,
      numTokens,
      '_prefill'
    );
    if (isTraceEnabled('embed')) {
      const overrideOffset = embeddingOverride?.offset ?? null;
      const overrideEnd = embeddingOverride
        ? embeddingOverride.offset + embeddingOverride.prefixLength - 1
        : null;
      trace.embed(
        `Prefill embedding IDs: tokens=${numTokens}, first=${embeddingInputIds[0] ?? 'missing'}, ` +
        `last=${embeddingInputIds[numTokens - 1] ?? 'missing'}, overrideOffset=${overrideOffset ?? 'none'}, ` +
        `overrideEnd=${overrideEnd ?? 'none'}, ` +
        `offsetId=${overrideOffset == null ? 'none' : embeddingInputIds[overrideOffset]}, ` +
        `afterOverrideId=${overrideEnd == null ? 'none' : embeddingInputIds[overrideEnd + 1] ?? 'missing'}`
      );
    }
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.prefillProfileSteps = [];

    if (startPos === 0 && hasLinearAttentionLayers(config.layerTypes)) {
      this._state.linearAttentionRuntime = resetLinearAttentionRuntime(this._state.linearAttentionRuntime);
    }
    if (startPos === 0) {
      for (const [, convState] of this._state.convLayerStates) {
        if (convState.convStateGPU && convState.hiddenSize && convState.kernelSize) {
          uploadData(convState.convStateGPU, new Float32Array(convState.hiddenSize * (convState.kernelSize - 1)));
        }
      }
    }

    const embedBufferRaw = this._state.weights.get('embed');
    if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !isSplitWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
      throw new Error('Embed buffer not found or not a supported buffer type');
    }
    const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
    const embedDtype = isCpuWeightBuffer(embedBufferRaw)
      ? embedBufferRaw.dtype
      : getWeightDtype(embedBufferRaw);
    const embedMetadata = getWeightMetadata(embedBufferRaw);
    if (opts.debug) {
      const embedSize = isGpuBufferInstance(embedBuffer) ? embedBuffer.size : 'N/A';
      log.debug('Pipeline', `Embed buffer: type=${embedBuffer?.constructor?.name}, size=${embedSize}, dtype=${embedDtype}`);
    }

    const device = getDevice();
    const useCheckpoints = opts.debugLayers && opts.debugLayers.length > 0;
    const disableCommandBatching = shouldDisablePrefillCommandBatching(
      this._state,
      opts,
      multimodalBidirectionalSpan
    );
    const createRecorder = (label) => {
      if (!device || disableCommandBatching) return undefined;
      const recorderOptions = {
        recordLabels: opts.debug === true || opts.benchmark === true || opts.executionObserver === true,
        recordDispatches: opts.debug === true || opts.executionObserver === true,
        aggregateDispatches: opts.executionObserver === true,
      };
      return opts.profile
        ? createProfilingRecorder(label, device, recorderOptions)
        : createCommandRecorder(label, recorderOptions, device);
    };
    const recorder = createRecorder('prefill');
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim)
      : undefined;
    const context = buildLayerContext(
      this._state,
      recorder,
      false,
      opts.debugLayers,
      debugCheckBuffer,
      opts.executionPlan
    );
    context.trainingCapture = opts?._trainingCapture ?? null;
    context.currentTokenIds = inputIds;
    context.diffusionGemmaDecoder = opts?._diffusionGemmaDecoder === true;
    context.skipKVCacheWrites = returnHidden
      && opts?._embeddingOnly === true
      && this._state.runtimeConfig?.inference?.session?.skipEmbeddingKVCacheWrites === true;
    context.multimodalBidirectionalSpan = multimodalBidirectionalSpan == null
      ? null
      : {
        start: startPos + multimodalBidirectionalSpan.offset,
        length: multimodalBidirectionalSpan.length,
      };
    let gpuTimePrefillMs = 0;
    let hasGpuTimePrefill = false;
    const recordProfile = async (rec) => {
      if (!opts.profile || !rec) return;
      const timings = rec.isProfilingEnabled()
        ? await rec.resolveProfileTimings()
        : null;
      const total = sumProfileTimings(timings);
      if (total !== null) {
        gpuTimePrefillMs += total;
        hasGpuTimePrefill = true;
      }
      if (timings || opts.executionObserver) {
        recordPrefillProfileStep(this._state, {
          label: rec.label,
          timings,
          totalMs: total ?? undefined,
          recorderStats: rec.getStats(),
        });
        if (!opts.executionObserver) {
          log.warn('Profile', `Prefill (${rec.label}):`);
          log.warn('Profile', CommandRecorder.formatProfileReport(timings));
        }
      }
    };

    const benchmarkSubmits = opts.debug;
    if (benchmarkSubmits) {
      setTrackSubmits(true);
      resetSubmitStats();
    }

    const preserveBufferAcrossRecorderSubmit = (buffer, activeRecorder, label) => {
      if (!activeRecorder || !isGpuBufferInstance(buffer)) {
        return buffer;
      }
      const carryBuffer = acquireBuffer(
        buffer.size,
        typeof buffer.usage === 'number' ? buffer.usage : undefined,
        label
      );
      activeRecorder.getEncoder().copyBufferToBuffer(buffer, 0, carryBuffer, 0, buffer.size);
      activeRecorder.trackTemporaryBuffer(buffer);
      return carryBuffer;
    };

    const activationDtype = opts.executionPlan?.activationDtype ?? this._getEffectiveActivationDtype();
    const activationBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
    if (tracePrefillEnabled) {
      trace.perf('Prefill phase start', {
        numTokens,
        startPos,
        numLayers: config.numLayers,
        activationDtype,
        returnHidden,
      });
    }
    const embedTraceStart = tracePrefillEnabled ? performance.now() : 0;
    let baseEmbeddings = await embed(embeddingInputIds, embedBuffer, {
      hiddenSize: config.hiddenSize,
      vocabSize: config.vocabSize,
      scaleEmbeddings: config.scaleEmbeddings,
      embeddingScale: config.embeddingScale, embeddingNormalization: config.embeddingNormalization,
      debug: opts.debug,
      recorder,
      transpose: this._state.embeddingTranspose,
      debugProbes: this._state.runtimeConfig.shared.debug.probes,
      operatorDiagnostics: this._state.operatorDiagnostics,
      activationDtype,
      embeddingDtype: selectRuleValue('inference', 'dtype', 'embeddingDtype', { dtype: embedDtype }),
      embeddingStorageEncoding: embedMetadata?.storageEncoding ?? null,
      executionPolicies: this._state.executionV1State?.policies ?? null,
    });
    if (tracePrefillEnabled) {
      trace.perf('Prefill embed complete', {
        numTokens,
        elapsedMs: performance.now() - embedTraceStart,
      });
    }
    await runProbes('prefill_base_embedding_out', baseEmbeddings.buffer, {
      numTokens,
      hiddenSize: config.hiddenSize,
      probes: this._state.runtimeConfig.shared.debug.probes,
      recorder,
      operatorDiagnostics: this._state.operatorDiagnostics,
      dtype: baseEmbeddings.dtype,
    });
    let hiddenStates = baseEmbeddings;
    let perLayerInputs = null;
    const perLayerInputsTraceStart = tracePrefillEnabled ? performance.now() : 0;
    try {
      hiddenStates = await applyPrefixEmbeddingOverride(
        baseEmbeddings,
        embeddingOverride,
        config.hiddenSize,
        '_prefill',
        {
          executionPolicies: this._state.executionV1State?.policies ?? null,
          transitionDeclaredBy: resolvePrefixEmbeddingOverrideTransitionDeclaredBy(this._state.executionV1State),
        }
      );
      await runProbes('prefill_embedding_override_out', hiddenStates.buffer, {
        numTokens,
        hiddenSize: config.hiddenSize,
        probes: this._state.runtimeConfig.shared.debug.probes,
        recorder,
        operatorDiagnostics: this._state.operatorDiagnostics,
        dtype: hiddenStates.dtype,
      });
      perLayerInputs = await preparePerLayerInputs(
        embeddingInputIds,
        embeddingInputIds === inputIds ? hiddenStates : baseEmbeddings,
        context,
        {
          numTokens,
          pleCache: this._state.pleCache ?? null,
        }
      );
      if (tracePrefillEnabled) {
        trace.perf('Prefill per-layer inputs complete', {
          numTokens,
          elapsedMs: performance.now() - perLayerInputsTraceStart,
          materialized: Array.isArray(perLayerInputs),
        });
      }
    } catch (error) {
      if (isGpuBufferInstance(hiddenStates?.buffer)) {
        releaseBuffer(hiddenStates.buffer);
      }
      if (hiddenStates === baseEmbeddings) {
        baseEmbeddings = null;
      }
      hiddenStates = null;
      throw error;
    } finally {
      if (hiddenStates !== baseEmbeddings && isGpuBufferInstance(baseEmbeddings?.buffer)) {
        releaseBuffer(baseEmbeddings.buffer);
      }
      baseEmbeddings = null;
    }

    if (opts.debug && isGpuBufferInstance(hiddenStates)) {
      if (recorder) {
        hiddenStates = createTensor(
          preserveBufferAcrossRecorderSubmit(hiddenStates.buffer, recorder, 'prefill_embed_carry'),
          hiddenStates.dtype,
          hiddenStates.shape,
          hiddenStates.label
        );
        recordPrefillRecorderStats(this._state, recorder);
        await recorder.submitAndWait();
        await recordProfile(recorder);
      }
      const debugReadbackSize = this._state.runtimeConfig.shared.debug.pipeline.readbackSampleSize;
      const sample = await readBuffer(hiddenStates, Math.min(debugReadbackSize, hiddenStates.size));
      const f32 = decodeReadback(sample, activationDtype);
      const nanCount = f32.filter(x => !Number.isFinite(x)).length;
      let maxAbs = 0;
      for (let i = 0; i < f32.length; i++) {
        const abs = Math.abs(f32[i]);
        if (abs > maxAbs) maxAbs = abs;
      }
      const first8 = Array.from(f32).slice(0, 8).map(x => x.toFixed(4)).join(', ');
      log.debug('Pipeline', `After embed: buffer.label=${hiddenStates.label}, buffer.size=${hiddenStates.size}, maxAbs=${maxAbs.toFixed(4)}`);
      log.debug('Pipeline', `After embed first8=[${first8}], nan=${nanCount}/${f32.length}`);
    }

    if (opts.debug) {
      log.debug('Pipeline', `LAYER_LOOP_START: numLayers=${config.numLayers}, useGPU=${context.useGPU}`);
    }

    if (this._state.finitenessBuffer) {
      const device = getDevice();
      if (device) {
        device.queue.writeBuffer(this._state.finitenessBuffer, 0, FINITENESS_RESET_WORDS);
      }
    }

    let currentRecorder = recorder;

    // Chunked recorder submission: submit every N layers to release tracked intermediate
    // buffers, preventing unbounded memory growth during large prefills. Critical for
    // replay_prefill models where each decode step re-runs a prefill-style layer pass.
    const prefillRecorderChunkLayers = resolvePrefillRecorderChunkLayers({
      configuredPrefillChunkLayers: this._state.runtimeConfig?.inference?.session?.prefillChunkLayers,
      hasGpuSplitPerLayerInputs: context.perLayerInputsSession?.materialization === 'gpu_split_tables',
      numTokens,
    });

    let currentHiddenBuffer = hiddenStates.buffer;
    let prefillRecordMs = 0;
    let prefillSubmitWaitMs = 0;
    const layerLoopTraceStart = tracePrefillEnabled ? performance.now() : 0;
    const traceLayerHealthEnabled = isTraceEnabled('logits');
    try {
      for (let l = 0; l < config.numLayers; l++) {
        // Per-layer hard cancellation: when the caller's AbortSignal aborts,
        // exit the prefill loop between layer dispatches rather than
        // continuing to burn GPU on superseded work. Granularity is one
        // layer (~5-50ms), which is the fastest cancel granularity WebGPU
        // exposes today.
        if (opts?.signal?.aborted) {
          const reason = typeof opts.signal.reason === "string" ? opts.signal.reason : "Doppler: prefill aborted";
          const err = new Error(reason);
          err.name = "AbortError";
          err.code = "ABORT_ERR";
          throw err;
        }
        context.recorder = currentRecorder;
        context.perLayerInputBuffer = perLayerInputs?.[l] ?? null;

        const prevBuffer = currentHiddenBuffer;
        const layerRecordStart = performance.now();
        const layerOutput = await processLayer(l, currentHiddenBuffer, numTokens, true, context);
        prefillRecordMs += performance.now() - layerRecordStart;
        if (!isGpuBufferInstance(layerOutput)) throw new Error('Expected GPUBuffer from processLayer');
        currentHiddenBuffer = layerOutput;
        releasePerLayerInputBuffer(
          context.perLayerInputBuffer,
          currentRecorder,
          context.decodeBuffers,
          this._state.pleCache ?? null
        );
        if (perLayerInputs) {
          perLayerInputs[l] = null;
        }
        context.perLayerInputBuffer = null;

        const isCheckpoint = useCheckpoints && opts.debugLayers?.includes(l);
        const isChunkBoundary = !isCheckpoint
          && !traceLayerHealthEnabled
          && currentRecorder
          && l < config.numLayers - 1
          && (l + 1) % prefillRecorderChunkLayers === 0;

        if (isCheckpoint && currentRecorder) {
          currentHiddenBuffer = preserveBufferAcrossRecorderSubmit(
            currentHiddenBuffer,
            currentRecorder,
            'prefill_checkpoint_carry'
          );
          recordPrefillRecorderStats(this._state, currentRecorder);
          await currentRecorder.submitAndWait();
          await recordProfile(currentRecorder);
          currentRecorder = undefined;
        }

        if (traceLayerHealthEnabled && currentRecorder) {
          currentHiddenBuffer = preserveBufferAcrossRecorderSubmit(
            currentHiddenBuffer,
            currentRecorder,
            'prefill_trace_layer_health_carry'
          );
          const traceSubmitStart = performance.now();
          recordPrefillRecorderStats(this._state, currentRecorder);
          await currentRecorder.submitAndWait();
          await recordProfile(currentRecorder);
          prefillSubmitWaitMs += performance.now() - traceSubmitStart;
          await traceActivationHealth(
            `PREFILL_LAYER_${l}_HEALTH`,
            currentHiddenBuffer,
            activationDtype,
            numTokens * config.hiddenSize
          );
          currentRecorder = l < config.numLayers - 1
            ? createRecorder('prefill-trace')
            : undefined;
        }

        const shouldDebug = opts.debug && currentHiddenBuffer && (!recorder || isCheckpoint);
        if (shouldDebug && !currentRecorder) {
          const device = getDevice();
          if (device) {
            if (allowReadback(`pipeline.prefill.layer-${l}`)) {
              try {
                const sampleSize = config.hiddenSize * activationBytes;
                const lastTokenOffset = (numTokens - 1) * config.hiddenSize * activationBytes;
                const readback = await readBufferSlice(currentHiddenBuffer, lastTokenOffset, sampleSize);
                const data = decodeReadback(readback, activationDtype);
                let min = Infinity;
                let max = -Infinity;
                let maxAbs = 0;
                for (const v of data) {
                  if (!Number.isFinite(v)) continue;
                  if (v < min) min = v;
                  if (v > max) max = v;
                  const av = Math.abs(v);
                  if (av > maxAbs) maxAbs = av;
                }
                const sample = Array.from(data).slice(0, 3).map(x => x.toFixed(3)).join(', ');
                log.debug('Pipeline', `LAYER_${l}_LAST[pos=${numTokens - 1}]: min=${min.toFixed(3)}, max=${max.toFixed(3)}, maxAbs=${maxAbs.toFixed(2)}, sample=[${sample}]`);
              } catch (e) {
                log.debug('Pipeline', `LAYER_${l}_LAST: error reading buffer: ${e}`);
              }
            }
          }
        }

        if (isCheckpoint && useCheckpoints && l < config.numLayers - 1) {
          currentRecorder = createRecorder('prefill-cont');
        }

        if (prevBuffer !== currentHiddenBuffer) {
          if (currentRecorder) {
            currentRecorder.trackTemporaryBuffer(prevBuffer);
          } else {
            releaseBuffer(prevBuffer);
          }
        }

        if (isChunkBoundary) {
          currentHiddenBuffer = preserveBufferAcrossRecorderSubmit(
            currentHiddenBuffer,
            currentRecorder,
            'prefill_chunk_carry'
          );
          // Chunk boundary exists only to bound intermediate buffer lifetime.
          // When the runtime opts into async chunk submit AND profile timings
          // are not being collected, skip the CPU-GPU wait: queue order is
          // preserved across submits and deferred cleanup still releases
          // tracked buffers when GPU work completes. Profile runs keep the
          // sync path because resolveProfileTimings requires the submitted
          // work to be done.
          const chunkSubmitMode = resolvePrefillChunkSubmitMode(
            this._state.runtimeConfig,
            this._state.modelConfig
          );
          const chunkSubmitStart = performance.now();
          recordPrefillRecorderStats(this._state, currentRecorder);
          if (chunkSubmitMode === 'async' && !opts.profile) {
            currentRecorder.submit({ cleanup: 'deferred' });
          } else {
            await currentRecorder.submitAndWait();
            await recordProfile(currentRecorder);
          }
          prefillSubmitWaitMs += performance.now() - chunkSubmitStart;
          await traceActivationHealth(
            `PREFILL_LAYER_${l}_HEALTH`,
            currentHiddenBuffer,
            activationDtype,
            numTokens * config.hiddenSize
          );
          if (tracePrefillEnabled) {
            trace.perf('Prefill chunk submitted', {
              layer: l,
              elapsedMs: performance.now() - layerLoopTraceStart,
              prefillRecordMs,
              prefillSubmitWaitMs,
            });
          }
          currentRecorder = createRecorder('prefill-chunk');
        }
      }
      if (tracePrefillEnabled) {
        trace.perf('Prefill layer loop recorded', {
          numTokens,
          elapsedMs: performance.now() - layerLoopTraceStart,
          prefillRecordMs,
          prefillSubmitWaitMs,
        });
      }
    } finally {
      context.perLayerInputBuffer = null;
      if (perLayerInputs) {
        for (const buffer of perLayerInputs) {
          releasePerLayerInputBuffer(
            buffer,
            currentRecorder,
            context.decodeBuffers,
            this._state.pleCache ?? null
          );
        }
      }
      releaseSharedAttentionState(context.sharedAttentionState, currentRecorder);
      this._state.stats.prefillRecordMs = (this._state.stats.prefillRecordMs ?? 0) + prefillRecordMs;
      this._state.stats.prefillSubmitWaitMs = (this._state.stats.prefillSubmitWaitMs ?? 0) + prefillSubmitWaitMs;
    }

    {
      const tsirFixtureCfg = this._state.operatorDiagnostics?.tsirFixture ?? null;
      if (tsirFixtureCfg && Array.isArray(tsirFixtureCfg.pendingReads) && tsirFixtureCfg.pendingReads.length > 0) {
        if (currentRecorder) {
          currentHiddenBuffer = preserveBufferAcrossRecorderSubmit(
            currentHiddenBuffer,
            currentRecorder,
            'prefill_tsir_drain_carry'
          );
          recordPrefillRecorderStats(this._state, currentRecorder);
          await currentRecorder.submitAndWait();
          await recordProfile(currentRecorder);
          currentRecorder = undefined;
        }
        await drainPendingTsirReads(tsirFixtureCfg);
      }
    }
    if (this._state.finitenessBuffer) {
      if (currentRecorder) {
        currentHiddenBuffer = preserveBufferAcrossRecorderSubmit(
            currentHiddenBuffer,
            currentRecorder,
            'prefill_finiteness_carry'
          );
          recordPrefillRecorderStats(this._state, currentRecorder);
          await currentRecorder.submitAndWait();
          await recordProfile(currentRecorder);
          currentRecorder = undefined;
      }
      const isInfiniteData = await readBuffer(this._state.finitenessBuffer, 16);
      const u32 = new Uint32Array(isInfiniteData.buffer, isInfiniteData.byteOffset, 4);
      const finitenessStatus = parseFinitenessStatusWords(u32, 0);
      if (finitenessStatus.triggered) {
        if (isGpuBufferInstance(currentHiddenBuffer)) {
          releaseBuffer(currentHiddenBuffer);
        }
        throw new FinitenessError(`F16 bounds exceeded during prefill${finitenessStatus.metadata}`);
      }
    }

    if (benchmarkSubmits) {
      logSubmitStats(`Prefill (${numTokens} tokens, ${config.numLayers} layers)`);
      setTrackSubmits(false);
    }

    if (opts.debug) {
      log.debug('Pipeline', `LAYER_LOOP_DONE, currentHiddenBuffer type=${currentHiddenBuffer?.constructor?.name}`);
      if (currentHiddenBuffer && allowReadback('pipeline.prefill.final-hidden')) {
        const lastTokenOffset = (numTokens - 1) * config.hiddenSize * activationBytes;
        const sampleSize = config.hiddenSize * activationBytes;
        const data = decodeReadback(
          await readBufferSlice(currentHiddenBuffer, lastTokenOffset, sampleSize),
          activationDtype
        );
        const nanCount = Array.from(data).filter(x => !Number.isFinite(x)).length;
        const nonZero = Array.from(data).filter(x => Number.isFinite(x) && x !== 0).slice(0, 5);
        log.debug('Pipeline', `FINAL_HIDDEN[pos=${numTokens - 1}]: nan=${nanCount}/${data.length}, sample=[${nonZero.map(x => x.toFixed(4)).join(', ')}]`);
      }
    }

    if (hasGpuTimePrefill) {
      this._state.stats.gpuTimePrefillMs = gpuTimePrefillMs;
    }

    if (returnHidden) {
      if (currentRecorder) {
        currentHiddenBuffer = preserveBufferAcrossRecorderSubmit(
          currentHiddenBuffer,
          currentRecorder,
          'prefill_return_hidden_carry'
        );
      }
      return {
        numTokens,
        config,
        startPos,
        activationDtype,
        activationBytes,
        currentRecorder,
        recordProfile,
        debugCheckBuffer,
        currentHiddenBuffer,
      };
    }

    let lastLogits;
    let logitsVocabSize = config.vocabSize;
    const lmHead = this._state.weights.get('lm_head');
    const canRecordLogits = !!currentRecorder
      && selectedLogitTokenIds === null
      && !!lmHead
      && !isCpuWeightBuffer(lmHead)
      && !this._state.disableRecordedLogits
      && numTokens === 1;
    if (currentRecorder && canRecordLogits) {
      const logitsTraceStart = tracePrefillEnabled ? performance.now() : 0;
      const recorded = await recordLogitsGPU(
        currentRecorder,
        currentHiddenBuffer,
        numTokens,
        getLogitsWeights(this._state),
        getLogitsConfig(this._state),
        this._state.operatorDiagnostics,
        { applySoftcap: true }
      );
      logitsVocabSize = recorded.vocabSize;

      recordPrefillRecorderStats(this._state, currentRecorder);
      await currentRecorder.submitAndWait();
      await recordProfile(currentRecorder);

      const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: recorded.logitsDtype });
      const lastLogitsSize = logitsVocabSize * logitsBytes;
      const lastLogitsOffset = (numTokens - 1) * lastLogitsSize;
      const logitsData = await readBufferSlice(recorded.logitsBuffer, lastLogitsOffset, lastLogitsSize);
      releaseBuffer(recorded.logitsBuffer);
      lastLogits = decodeReadback(logitsData, recorded.logitsDtype);
      if (tracePrefillEnabled) {
        trace.perf('Prefill recorded logits complete', {
          numTokens,
          vocabSize: logitsVocabSize,
          elapsedMs: performance.now() - logitsTraceStart,
        });
      }

      const health = getLogitsHealth(lastLogits);
      if (health.nanCount > 0 || health.infCount > 0 || health.nonZeroCount === 0) {
        log.warn(
          'Logits',
          `Recorded logits invalid (nan=${health.nanCount} inf=${health.infCount} nonZero=${health.nonZeroCount}, maxAbs=${health.maxAbs.toFixed(3)}); recomputing without recorder.`
        );
        this._state.disableRecordedLogits = true;
        this._state.disableFusedDecode = true;
        const fallbackLogits = await computeLogits(
          currentHiddenBuffer,
          numTokens,
          getLogitsWeights(this._state),
          getLogitsConfig(this._state),
          this._state.useGPU,
          this._state.debugFlags,
          undefined,
          debugCheckBuffer,
          this._state.runtimeConfig.shared.debug.probes,
          { lastPositionOnly: true },
          this._state.operatorDiagnostics
        );
        const fallbackHealth = getLogitsHealth(fallbackLogits);
        if (fallbackHealth.nanCount > 0 || fallbackHealth.infCount > 0 || fallbackHealth.nonZeroCount === 0) {
          throw new Error(
            `[Logits] Fallback logits invalid (nan=${fallbackHealth.nanCount} inf=${fallbackHealth.infCount} nonZero=${fallbackHealth.nonZeroCount}, maxAbs=${fallbackHealth.maxAbs.toFixed(3)}). ` +
            'This indicates upstream kernel output is NaN/Inf (often prefill attention/matmul).'
          );
        }
        logitsVocabSize = config.vocabSize;
        lastLogits = fallbackLogits.length === logitsVocabSize
          ? fallbackLogits
          : extractLastPositionLogits(fallbackLogits, numTokens, logitsVocabSize);
      }

      releaseBuffer(currentHiddenBuffer);
    } else {
      const logitsTraceStart = tracePrefillEnabled ? performance.now() : 0;
      if (currentRecorder) {
        currentHiddenBuffer = preserveBufferAcrossRecorderSubmit(
          currentHiddenBuffer,
          currentRecorder,
          'prefill_logits_carry'
        );
        recordPrefillRecorderStats(this._state, currentRecorder);
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
      }
      const logits = await computeLogits(
        currentHiddenBuffer,
        numTokens,
        getLogitsWeights(this._state),
        getLogitsConfig(this._state),
        this._state.useGPU,
        this._state.debugFlags,
        undefined,
        debugCheckBuffer,
        this._state.runtimeConfig.shared.debug.probes,
        { lastPositionOnly: true, selectedTokenIds: selectedLogitTokenIds },
        this._state.operatorDiagnostics
      );

      if (selectedLogitTokenIds !== null) {
        lastLogits = logits;
      } else {
        lastLogits = logits.length === logitsVocabSize
          ? logits
          : extractLastPositionLogits(logits, numTokens, logitsVocabSize);
      }
      releaseBuffer(currentHiddenBuffer);
      if (tracePrefillEnabled) {
        trace.perf('Prefill logits complete', {
          numTokens,
          vocabSize: logitsVocabSize,
          elapsedMs: performance.now() - logitsTraceStart,
        });
      }
    }

    this._state.currentSeqLen = startPos + numTokens;

    if (opts.debug && selectedLogitTokenIds === null) {
      logitsSanity(lastLogits, 'Prefill', (tokens) => resolveTokenText(this._state.tokenizer, tokens));
    }
    if (isTraceEnabled('logits')) {
      trace.logits(selectedLogitTokenIds === null ? 'PREFILL_LOGITS_HEALTH' : 'PREFILL_SELECTED_LOGITS_HEALTH', getLogitsHealth(lastLogits));
    }

    if (opts.debug) {
      if (this._state.kvCache?.hasGPUCache?.()) {
        log.debug('Pipeline', `KV cache active after prefill: seqLen=${this._state.kvCache.getKeyCache(0)?.constructor.name ?? '?'}`);
      } else {
        log.warn('Pipeline', `KV cache NOT active after prefill! hasGPUCache=${this._state.kvCache?.hasGPUCache?.()}`);
      }
    }

    if (tracePrefillEnabled) {
      trace.perf('Prefill phase complete', {
        numTokens,
        totalMs: performance.now() - prefillTraceStart,
      });
    }

    return lastLogits;
  }
