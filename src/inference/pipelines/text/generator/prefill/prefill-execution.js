import { computePrefillLogits } from "./prefill-logits.js";
import { getDevice, setTrackSubmits } from "../../../../../gpu/device.js";
import { acquireBuffer, releaseBuffer, readBuffer, readBufferSlice, uploadData, getBufferPool } from "../../../../../memory/buffer-pool.js";
import { resetSubmitStats, logSubmitStats } from "../../../../../gpu/submit-tracker.js";
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from "../../../../../gpu/command-recorder.js";
import { allowReadback } from "../../../../../debug/perf-guards.js";
import { log, trace, isTraceEnabled } from "../../../../../debug/index.js";
import { selectRuleValue } from "../../../../../rules/rule-registry.js";
import { logitsSanity } from "../sampling/sampling.js";
import { embed } from "../../embedding/embed.js";
import { processLayer } from "../../execution/layer.js";
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer, getWeightDtype, getWeightMetadata } from "../../../../../gpu/weight-buffer.js";
import { sumProfileTimings, FinitenessError } from "../generator-steps.js";
import { buildLayerContext, debugCheckBuffer as debugCheckBufferHelper } from "../session/session-context.js";
import { releaseSharedAttentionState } from "../session/attention-lifecycle.js";
import { decodeReadback } from "../../debug-utils/readback-decoding.js";
import { getLogitsHealth } from "../../debug-utils/health-metrics.js";
import { parseFinitenessStatusWords } from "../../policy/finiteness-guard-status.js";
import { drainPendingTsirReads } from "../../observation/tsir-fixture-writer.js";
import { resolvePrefillRecorderChunkLayers } from "../../execution/execution-plan.js";
import { hasLinearAttentionLayers, resetLinearAttentionSequence } from "../../linear-attention/linear-attention.js";
import { preparePerLayerInputs } from "../../per-layer/per-layer-inputs.js";
import { getActivationRepresentation } from "../session/activation-storage.js";
import { createTensor, tensorElementBytes } from "../../../../../gpu/tensor.js";
import { releasePerLayerInputBuffer, resolvePrefillMaxInFlightChunks, shouldDisablePrefillCommandBatching } from "./prefill-policy.js";
import { logPrefillCacheState, probePrefillEmbedding, tracePrefillEmbeddingIds } from "./prefill-observation.js";
import { normalizePrefixEmbeddingOverride, resolvePrefillEmbeddingInputIds, resolvePrefillMultimodalBidirectionalSpan, applyPrefixEmbeddingOverride, resolvePrefixEmbeddingOverrideTransitionDeclaredBy } from "./prefix-embedding.js";
import { recordPrefillProfileStep, resolveTokenText } from "../../policy/generator-decode-policy.js";
import { FINITENESS_RESET_WORDS, normalizeSelectedLogitTokenIds, recordPrefillRecorderStats, resolvePrefillChunkSubmitMode, traceActivationHealth } from "../session/text.js";

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
    tracePrefillEmbeddingIds(embeddingInputIds, numTokens, embeddingOverride);
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.prefillProfileSteps = [];

    if (startPos === 0 && hasLinearAttentionLayers(config.layerTypes)) {
      this._state.linearAttentionRuntime = resetLinearAttentionSequence(this._state.linearAttentionRuntime);
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
    const bufferPool = device ? getBufferPool(device) : null;
    const useCheckpoints = opts.debugLayers && opts.debugLayers.length > 0;
    const disableCommandBatching = shouldDisablePrefillCommandBatching(
      this._state,
      opts,
      multimodalBidirectionalSpan,
      { startPos, returnHidden }
    );
    if (this._state.dopplerLoader?.getRoutedExpertPolicy?.()?.enabled === true) log.debug('Prefill',
      'Command recording disabled: paged experts support segmented recording only in single-token decode.');
    const createRecorder = (label) => {
      if (!device || disableCommandBatching) return undefined;
      const recorderOptions = {
        recordLabels: opts.debug === true || opts.benchmark === true || opts.executionObserver === true,
        recordDispatches: opts.debug === true || opts.executionObserver === true,
        aggregateDispatches: opts.executionObserver === true,
      };
      return opts.profile
        ? createProfilingRecorder(label, device, { ...recorderOptions, profilePhase: 'prefill' })
        : createCommandRecorder(label, recorderOptions, device);
    };
    const recorder = createRecorder('prefill');
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;
    const context = buildLayerContext(
      this._state, recorder, false, opts.debugLayers, debugCheckBuffer, opts.executionPlan
    );
    context.trainingCapture = opts?._trainingCapture ?? null;
    context.currentTokenIds = inputIds;
    opts.__activationTrace?.activity?.begin('prefill', startPos, inputIds);
    context.signal = opts.signal;
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
      const timings = await rec.resolveProfileTimings({ requireComplete: true });
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
    const representation = getActivationRepresentation(this._state, opts.executionPlan);
    const activationBytes = tensorElementBytes(activationDtype, representation);
    if (representation.precision && embeddingOverride) {
      throw new Error('[Pipeline] Paired activations do not support prefix embedding overrides.');
    }
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
    let baseEmbeddings = null;
    let hiddenStates = null;
    let perLayerInputs = null;
    try {
      baseEmbeddings = await embed(embeddingInputIds, embedBuffer, {
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
        intermediatePrecision: representation.precision,
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
      hiddenStates = baseEmbeddings;
      const perLayerInputsTraceStart = tracePrefillEnabled ? performance.now() : 0;
      await probePrefillEmbedding('prefill_base_embedding_out', baseEmbeddings, {
        numTokens,
        hiddenSize: config.hiddenSize,
        state: this._state,
        recorder,
      });
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
      await probePrefillEmbedding('prefill_embedding_override_out', hiddenStates, {
        numTokens,
        hiddenSize: config.hiddenSize,
        state: this._state,
        recorder,
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
      const cleanupErrors = [];
      try { await recorder?.abort(); } catch (cleanupError) { cleanupErrors.push(cleanupError); }
      try { await bufferPool?.settleDeferredDestruction(); } catch (cleanupError) { cleanupErrors.push(cleanupError); }
      if (cleanupErrors.length) throw new AggregateError([error, ...cleanupErrors], 'Prefill preparation cleanup failed.', { cause: error });
      throw error;
    } finally {
      if (hiddenStates !== baseEmbeddings && isGpuBufferInstance(baseEmbeddings?.buffer)) {
        releaseBuffer(baseEmbeddings.buffer);
      }
      baseEmbeddings = null;
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
    const submittedChunks = [];
    const settleSubmittedChunks = async () => {
      const results = await Promise.allSettled(submittedChunks.splice(0).map(
        (chunk) => chunk.completeDeferredCleanup()
      ));
      const errors = results.filter((result) => result.status === 'rejected').map((result) => result.reason);
      if (errors.length === 1) throw errors[0];
      if (errors.length > 1) throw new AggregateError(errors, 'Prefill chunk completion failed.', { cause: errors[0] });
    };

    // Chunked recorder submission: submit every N layers to release tracked intermediate
    // buffers, preventing unbounded memory growth during large prefills. Critical for
    // replay-prefill models where each decode step re-runs a prefill-style layer pass.
    const prefillRecorderChunkLayers = resolvePrefillRecorderChunkLayers({
      configuredPrefillChunkLayers: this._state.runtimeConfig?.inference?.session?.prefillChunkLayers,
      hasGpuSplitPerLayerInputs: context.perLayerInputsSession?.materialization === 'gpu-split-tables',
      numTokens,
    });

    const residualWidth = config.hyperConnections ? config.hyperConnections.streamCount * config.hiddenSize : config.hiddenSize;
    const prefillBuffers = { hidden: hiddenStates.buffer };
    let transferred = false;
    let failed = false;
    let prefillError;
    try {
      let prefillRecordMs = 0;
      let prefillSubmitWaitMs = 0;
      const layerLoopTraceStart = tracePrefillEnabled ? performance.now() : 0;
      const traceLayerHealthEnabled = isTraceEnabled('logits');
      try {
        if (opts.debug && isGpuBufferInstance(hiddenStates?.buffer)) {
          if (recorder) {
            hiddenStates = createTensor(
              preserveBufferAcrossRecorderSubmit(hiddenStates.buffer, recorder, 'prefill_embed_carry'),
              hiddenStates.dtype,
              hiddenStates.shape,
              hiddenStates.label,
              hiddenStates
            );
            prefillBuffers.hidden = hiddenStates.buffer;
            recordPrefillRecorderStats(this._state, recorder);
            await recorder.submitAndWait();
            await recordProfile(recorder);
          }
          const debugReadbackSize = this._state.runtimeConfig.shared.debug.pipeline.readbackSampleSize;
          const sampleSize = Math.floor(Math.min(debugReadbackSize, hiddenStates.buffer.size) / activationBytes) * activationBytes;
          const sample = await readBuffer(hiddenStates.buffer, sampleSize);
          const f32 = decodeReadback(sample, activationDtype, representation);
          const nanCount = f32.filter(x => !Number.isFinite(x)).length;
          let maxAbs = 0;
          for (let i = 0; i < f32.length; i++) {
            const abs = Math.abs(f32[i]);
            if (abs > maxAbs) maxAbs = abs;
          }
          const first8 = Array.from(f32).slice(0, 8).map(x => x.toFixed(4)).join(', ');
          log.debug('Pipeline', `After embed: buffer.label=${hiddenStates.label}, buffer.size=${hiddenStates.buffer.size}, maxAbs=${maxAbs.toFixed(4)}`);
          log.debug('Pipeline', `After embed first8=[${first8}], nan=${nanCount}/${f32.length}`);
        }
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

          const prevBuffer = prefillBuffers.hidden;
          const layerRecordStart = performance.now();
          const layerOutput = await processLayer(l, prefillBuffers.hidden, numTokens, true, context);
          prefillRecordMs += performance.now() - layerRecordStart;
          if (!isGpuBufferInstance(layerOutput)) throw new Error('Expected GPUBuffer from processLayer');
          prefillBuffers.hidden = layerOutput;
          opts.__activationTrace?.capture(layerOutput, { layer: l, phase: 'prefill', tokenStart: startPos,
            tokenIds: inputIds, dtype: activationDtype, recorder: currentRecorder, ...representation });
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
            prefillBuffers.hidden = preserveBufferAcrossRecorderSubmit(
              prefillBuffers.hidden,
              currentRecorder,
              'prefill_checkpoint_carry'
            );
            recordPrefillRecorderStats(this._state, currentRecorder);
            await currentRecorder.submitAndWait();
            await recordProfile(currentRecorder);
            currentRecorder = undefined;
          }

          if (traceLayerHealthEnabled && currentRecorder) {
            prefillBuffers.hidden = preserveBufferAcrossRecorderSubmit(
              prefillBuffers.hidden,
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
              prefillBuffers.hidden,
              activationDtype,
              numTokens * residualWidth, representation
            );
            currentRecorder = l < config.numLayers - 1
              ? createRecorder('prefill-trace')
              : undefined;
          }

          const shouldDebug = opts.debug && prefillBuffers.hidden && (!recorder || isCheckpoint);
          if (shouldDebug && !currentRecorder) {
            const device = getDevice();
            if (device) {
              if (allowReadback(`pipeline.prefill.layer-${l}`)) {
                try {
                  const sampleSize = residualWidth * activationBytes;
                  const lastTokenOffset = (numTokens - 1) * residualWidth * activationBytes;
                  const readback = await readBufferSlice(prefillBuffers.hidden, lastTokenOffset, sampleSize);
                  const data = decodeReadback(readback, activationDtype, representation);
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

          if (prevBuffer !== prefillBuffers.hidden) {
            if (currentRecorder) {
              currentRecorder.trackTemporaryBuffer(prevBuffer);
            } else {
              releaseBuffer(prevBuffer);
            }
          }

          if (isChunkBoundary) {
            prefillBuffers.hidden = preserveBufferAcrossRecorderSubmit(
              prefillBuffers.hidden,
              currentRecorder,
              'prefill_chunk_carry'
            );
            // Chunk boundary exists only to bound intermediate buffer lifetime.
            // When the runtime opts into async chunk submit AND profile timings
            // are not being collected, overlap within the runtime queue bound: queue order is
            // preserved across submits and queue-owned cleanup releases
            // tracked buffers when GPU work completes. Retain each recorder
            // until completion to observe failures. Profile runs keep the
            // sync path because resolveProfileTimings requires the submitted
            // work to be done.
            const chunkSubmitMode = resolvePrefillChunkSubmitMode(this._state.runtimeConfig, this._state.modelConfig);
            const chunkSubmitStart = performance.now();
            recordPrefillRecorderStats(this._state, currentRecorder);
            if (chunkSubmitMode === 'async' && !opts.profile) {
              const maxInFlightChunks = resolvePrefillMaxInFlightChunks(this._state);
              currentRecorder.submit({ cleanup: 'queue' });
              submittedChunks.push(currentRecorder);
              if (submittedChunks.length >= maxInFlightChunks) {
                await submittedChunks[0].completeDeferredCleanup();
                submittedChunks.shift();
              }
            } else {
              await currentRecorder.submitAndWait();
              await recordProfile(currentRecorder);
            }
            prefillSubmitWaitMs += performance.now() - chunkSubmitStart;
            await traceActivationHealth(
              `PREFILL_LAYER_${l}_HEALTH`,
              prefillBuffers.hidden,
              activationDtype,
              numTokens * residualWidth, representation
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
            prefillBuffers.hidden = preserveBufferAcrossRecorderSubmit(
              prefillBuffers.hidden,
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
      // The prefill's last recorded commands: while the GPU runs them the caller may prepare
      // the step after the prompt (opts._whileFinalPrefillRuns, internal, at most once).
      const submitFinalRecorder = async (recorder) => {
        const settled = recorder.submitAndWait();
        try { await opts?._whileFinalPrefillRuns?.(startPos + numTokens, inputIds[numTokens - 1]); } finally { await settled; }
      };
      if (this._state.finitenessBuffer) {
        if (currentRecorder) {
          prefillBuffers.hidden = preserveBufferAcrossRecorderSubmit(
              prefillBuffers.hidden,
              currentRecorder,
              'prefill_finiteness_carry'
            );
            recordPrefillRecorderStats(this._state, currentRecorder);
            await submitFinalRecorder(currentRecorder);
            await recordProfile(currentRecorder);
            currentRecorder = undefined;
        }
        const isInfiniteData = await readBuffer(this._state.finitenessBuffer, 16);
        const u32 = new Uint32Array(isInfiniteData.buffer, isInfiniteData.byteOffset, 4);
        const finitenessStatus = parseFinitenessStatusWords(u32, 0);
        if (finitenessStatus.triggered) {
          throw new FinitenessError(`F16 bounds exceeded during prefill${finitenessStatus.metadata}`);
        }
      }

      if (benchmarkSubmits) {
        logSubmitStats(`Prefill (${numTokens} tokens, ${config.numLayers} layers)`);
        setTrackSubmits(false);
      }

      if (opts.debug) {
        log.debug('Pipeline', `LAYER_LOOP_DONE, currentHiddenBuffer type=${prefillBuffers.hidden?.constructor?.name}`);
        if (prefillBuffers.hidden && allowReadback('pipeline.prefill.final-hidden')) {
          const lastTokenOffset = (numTokens - 1) * residualWidth * activationBytes;
          const sampleSize = residualWidth * activationBytes;
          const data = decodeReadback(
            await readBufferSlice(prefillBuffers.hidden, lastTokenOffset, sampleSize),
            activationDtype, representation
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
        await settleSubmittedChunks();
        if (currentRecorder) {
          prefillBuffers.hidden = preserveBufferAcrossRecorderSubmit(
            prefillBuffers.hidden,
            currentRecorder,
            'prefill_return_hidden_carry'
          );
        }
        transferred = true;
        return {
          numTokens,
          config,
          startPos,
          activationDtype,
          activationBytes,
          representation,
          currentRecorder,
          recordProfile,
          debugCheckBuffer,
          currentHiddenBuffer: prefillBuffers.hidden,
        };
      }

      const lastLogits = await computePrefillLogits({
        config,
        currentRecorder,
        opts,
        selectedLogitTokenIds,
        numTokens,
        tracePrefillEnabled,
        prefillBuffers,
        recordProfile,
        debugCheckBuffer,
        preserveBufferAcrossRecorderSubmit,
        submitFinalRecorder,
        state: this._state
      });

      this._state.currentSeqLen = startPos + numTokens;

      if (opts._returnGpuLogits === true) return lastLogits;

      if (opts.debug && selectedLogitTokenIds === null) {
        logitsSanity(lastLogits, 'Prefill', (tokens) => resolveTokenText(this._state.tokenizer, tokens));
      }
      if (isTraceEnabled('logits')) {
        trace.logits(selectedLogitTokenIds === null ? 'PREFILL_LOGITS_HEALTH' : 'PREFILL_SELECTED_LOGITS_HEALTH', getLogitsHealth(lastLogits));
      }

      if (opts.debug) {
        logPrefillCacheState(this._state);
      }

      if (tracePrefillEnabled) {
        trace.perf('Prefill phase complete', {
          numTokens,
          totalMs: performance.now() - prefillTraceStart,
        });
      }

      return lastLogits;
    } catch (error) {
      failed = true;
      prefillError = error;
      throw error;
    } finally {
      if (!transferred) {
        const cleanupErrors = [];
        try { await settleSubmittedChunks(); } catch (error) { cleanupErrors.push(error); }
        try { releaseBuffer(prefillBuffers.hidden); } catch (error) { cleanupErrors.push(error); }
        try { await currentRecorder?.abort(); } catch (error) { cleanupErrors.push(error); }
        if (failed || cleanupErrors.length > 0) {
          try { await bufferPool?.settleDeferredDestruction(); } catch (error) { cleanupErrors.push(error); }
        }
        if (cleanupErrors.length > 0) {
          const errors = failed ? [prefillError, ...cleanupErrors] : cleanupErrors;
          if (errors.length === 1) throw errors[0];
          throw new AggregateError(errors, 'Prefill resource cleanup failed.', { cause: errors[0] });
        }
      }
    }
  }
