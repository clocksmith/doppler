import { lookupDraft, verifySpeculativeTokens } from "../session/speculative-runtime.js";
import { markDiagnosticSequenceStep } from "../session/diagnostic-sequence.js";
import { isGPUSamplingAvailable } from "../../../../../gpu/kernels/sample.js";
import { samplingSeedForToken } from "../../../../token-sampling.js";
import { matchesStopSequence } from "../sampling/stopping.js";
import { markWarmed as markKernelCacheWarmed } from "../../../../../gpu/kernel-selection-cache.js";
import { log } from "../../../../../debug/index.js";
import { validateCallTimeOptions } from "../../../../../config/param-validator.js";
import { selectRuleValue } from "../../../../../rules/rule-registry.js";
import { isStopToken } from "../../init.js";
import { isCpuWeightBuffer } from "../../../../../gpu/weight-buffer.js";
import { decodeStep, advanceWithToken as runAdvanceWithToken, generateNTokensGPU, shouldUseBatchDecode, advanceWithTokenAndEmbedding as runAdvanceWithTokenAndEmbedding } from "../generator-steps.js";
import { debugCheckBuffer as debugCheckBufferHelper, resolvePerLayerInputsSession } from "../session/session-context.js";
import { resolveAdvanceEmbeddingMode } from "../session/generator-runtime.js";
import { resetActiveExecutionPlan, resolveMaxBatchDecodeTokens, resolveActiveExecutionPlan } from "../../execution/execution-plan.js";
import { hasLinearAttentionLayers } from "../../linear-attention/linear-attention.js";
import { createPleBufferCache, prefetchPerLayerRow, getPleHotVocabularyRuntime, hasGpuSplitPerLayerInputEmbeddings, hasRangeBackedPerLayerInputEmbeddings } from "../../per-layer/per-layer-inputs.js";
import { shouldDisableBatchDecodeAfterShortBatch, resolveHotVocabularyBatchDecodeAvailability, primePleDecodeRuntimeCache, usesReplayPrefillDecode, assertIncrementalDecodeSupport } from "../../policy/generator-decode-policy.js";
import { assertBatchRetryInPlaceAllowed } from "./batch-retry-policy.js";
import { wholeTokenProgramRequested, prepareWholeTokenBatch } from "../whole-token/whole-token-lane.js";

export async function* _runDecodeLoop(generatedIds, opts, options, runtime) {
    const {
      stopTokenIds,
      eosToken,
      stopSequenceStart,
      decodeToken,
      logBatchPath = false,
      emitMode = 'text',
    } = runtime;

    let tokensGenerated = 1;
    markKernelCacheWarmed();

    // Step 4: Lazily initialise PLE buffer cache for decode-path slice reuse.
    const pleHiddenSize = Number(this._state.modelConfig.hiddenSizePerLayerInput ?? 0);
    if (pleHiddenSize > 0 && !this._state.pleCache) {
      const activationDtype = resolveActiveExecutionPlan(this._state).activationDtype;
      const bpe = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
      this._state.pleCache = createPleBufferCache(
        this._state.modelConfig.numLayers,
        pleHiddenSize * bpe,
      );
    }
    if (pleHiddenSize > 0) {
      await primePleDecodeRuntimeCache(this._state, generatedIds);
    }

    const decodeStart = performance.now();
    const resolvedPerLayerInputsSession = resolvePerLayerInputsSession(
      this._state.modelConfig.perLayerInputsSession ?? null,
      this._state.runtimeConfig?.inference?.session?.perLayerInputs ?? null
    );
    const lmHead = this._state.weights.get('lm_head');
    const embedBuffer = this._state.weights.get('embed');
    const hasCpuWeights = isCpuWeightBuffer(lmHead)
      || isCpuWeightBuffer(embedBuffer)
      || lmHead instanceof Float32Array
      || embedBuffer instanceof Float32Array;
    const hasLinearLayers = hasLinearAttentionLayers(this._state.modelConfig.layerTypes);
    const replayPrefillDecode = usesReplayPrefillDecode(this._state);
    const gpuSamplingAvailable = isGPUSamplingAvailable() && !hasCpuWeights;
    const hasSuppressedSamplingTokens = Array.isArray(opts.suppressTokenIds) && opts.suppressTokenIds.length > 0;
    const hasRangeBackedPerLayerInputs = hasRangeBackedPerLayerInputEmbeddings({
      config: this._state.modelConfig,
      weights: this._state.weights,
    });
    const hasGpuSplitPerLayerInputs = hasGpuSplitPerLayerInputEmbeddings({
      config: this._state.modelConfig,
      weights: this._state.weights,
      perLayerInputsSession: resolvedPerLayerInputsSession,
    });
    const pleHotVocabularyRuntime = getPleHotVocabularyRuntime({ weights: this._state.weights });
    const resolveCurrentHotVocabularyBatchDecodeAvailable = () => resolveHotVocabularyBatchDecodeAvailability({
      hasRangeBackedPerLayerInputs,
      pleHotVocabularyRuntime,
      tokenId: generatedIds[generatedIds.length - 1] ?? null,
    });
    const initialHotVocabularyBatchDecodeAvailable = resolveCurrentHotVocabularyBatchDecodeAvailable();
    const executionPlan = opts.executionPlan;
    const wholeTokenReplay = wholeTokenProgramRequested(this._state, opts);
    let useBatchPath = replayPrefillDecode || wholeTokenReplay
      ? false
        : shouldUseBatchDecode({
          batchSize: executionPlan.batchSize,
          useGPU: this._state.useGPU,
          gpuSamplingAvailable,
          disableMultiTokenDecode: executionPlan.disableMultiTokenDecode || hasSuppressedSamplingTokens || typeof opts.logitMaskFn === 'function',
          disableCommandBatching: executionPlan.disableCommandBatching,
          isBdpaPagedLayout: this._state.kvCache?.layout === 'bdpa_paged',
          finitenessFallbackWindowOpen: this._hasFinitenessFallbackWindow(),
          hasLinearAttentionLayers: hasLinearLayers,
          selfSpeculationEnabled: opts.speculation?.mode === 'self' && !initialHotVocabularyBatchDecodeAvailable,
          hasRangeBackedPerLayerInputs,
        });
    const ordinaryBatchPath = useBatchPath;
    if (opts.speculation?.mode === 'lookup') useBatchPath = false;
    if (!useBatchPath) {
      let reason = null;
      if (replayPrefillDecode) reason = 'replay_prefill_decode';
      else if (hasCpuWeights) reason = 'cpu_weights';
      else if (!this._state.useGPU) reason = 'no_gpu';
      else if (!gpuSamplingAvailable) reason = 'no_gpu_sampling';
      else if (executionPlan.disableCommandBatching) reason = 'command_batching_disabled';
      else if (hasSuppressedSamplingTokens) reason = 'sampling_suppression_requires_cpu_logits';
      else if (typeof opts.logitMaskFn === 'function') reason = 'token_constraint_requires_per_token_logits';
      else if (executionPlan.disableMultiTokenDecode) reason = 'multi_token_decode_disabled';
      else if (executionPlan.batchSize <= 1) reason = 'batch_size_1';
      else if (this._state.kvCache?.layout === 'bdpa_paged') reason = 'bdpa_paged_layout';
      else if (this._hasFinitenessFallbackWindow()) reason = 'finiteness_fallback_window';
      else if (hasLinearLayers) reason = 'linear_attention_layers';
      this._state.stats.decodeMode = replayPrefillDecode
        ? 'replay_prefill'
        : (opts.speculation?.mode === 'self' ? 'self_speculation' : 'single_token');
      this._state.stats.batchGuardReason = reason;
    } else {
      this._state.stats.decodeMode = hasRangeBackedPerLayerInputs
        ? 'batched_gpu_stepwise_ple'
        : 'batched_gpu';
      this._state.stats.batchGuardReason = null;
    }

    const readbackInterval = executionPlan.readbackInterval;
    const intervalBatches = readbackInterval == null ? 1 : readbackInterval;

    const decodeSingleTokenViaLogits = async () => this._decodeNextTokenViaLogits(generatedIds, opts);
    const stopIfAborted = () => {
      if (!options.signal?.aborted) return false;
      this._recordStopReason('aborted');
      return true;
    };

    if (logBatchPath && useBatchPath) {
      log.debug(
        'Pipeline',
        `Using batch decode path with batchSize=${executionPlan.batchSize}, stopCheckMode=${executionPlan.stopCheckMode}, readbackInterval=${readbackInterval}`
      );
    }

    const batchTokenLimit = (hotVocabularyBatchDecodeAvailable) => resolveMaxBatchDecodeTokens({
          hasHotVocabularyBatchDecode: hotVocabularyBatchDecodeAvailable,
          hasGpuSplitPerLayerInputs,
          hasLinearAttentionLayers: hasLinearLayers,
          modelId: this._state.modelConfig.modelId ?? executionPlan.kernelPathId,
          activationDtype: executionPlan.activationDtype,
          currentSeqLen: this._state.currentSeqLen,
          maxDecodeTokens: opts.maxTokens,
          numLayers: this._state.modelConfig.numLayers,
          hiddenSize: this._state.modelConfig.hiddenSize,
          configuredMaxBatchDecodeTokens: executionPlan.maxBatchDecodeTokens,
    });

    let rejectedDraftBlocks = 0;
    while (tokensGenerated < opts.maxTokens) {
      if (stopIfAborted()) break;
      if (this._hasFinitenessFallbackWindow() && useBatchPath) {
        useBatchPath = false;
      }
      const hotVocabularyBatchDecodeAvailable = resolveCurrentHotVocabularyBatchDecodeAvailable();

      if (opts.speculation?.mode === 'lookup' && rejectedDraftBlocks < opts.speculation.maxRejectedBlocks) {
        const remaining = opts.maxTokens - tokensGenerated;
        const proposal = remaining > 1 ? lookupDraft(generatedIds, Math.min(opts.speculation.tokens, remaining - 1), opts.speculation.maxMatchTokens, opts.speculation.minMatchTokens) : [];
        if (proposal.length) {
          const stop = (context, token) => isStopToken(token, stopTokenIds, eosToken)
            || matchesStopSequence(this._state.tokenizer, context, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput);
          const result = await verifySpeculativeTokens(this, generatedIds, proposal, opts, options.signal, stop);
          if (result.accepted < proposal.length) rejectedDraftBlocks++;
          if (rejectedDraftBlocks >= opts.speculation.maxRejectedBlocks) useBatchPath = ordinaryBatchPath;
          this._state.stats.decodeMode = 'target_verified_lookup';
          this._state.stats.speculationAttempts += proposal.length;
          this._state.stats.speculationAccepted += result.accepted;
          for (const token of result.tokens) {
            if (stopIfAborted()) break;
            generatedIds.push(token); tokensGenerated++;
            const text = emitMode === 'token' ? '' : decodeToken(token);
            yield emitMode === 'token' ? token : text;
            options.onToken?.(token, text);
          }
          const last = generatedIds[generatedIds.length - 1];
          if (isStopToken(last, stopTokenIds, eosToken)) { this._recordStopReason('stop-token', last); break; }
          if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
            this._recordStopReason('stop-sequence', last); break;
          }
          continue;
        }
      }

      if (wholeTokenReplay && executionPlan.batchSize > 1 && executionPlan.stopCheckMode === 'batch' && !executionPlan.disableMultiTokenDecode) {
        const requested = Math.min(executionPlan.batchSize, opts.maxTokens - tokensGenerated);
        const maximum = batchTokenLimit(hotVocabularyBatchDecodeAvailable);
        const bounded = maximum == null ? requested : Math.min(requested, maximum);
        if (maximum != null) this._state.batchingStats.maxBatchTokenCap = Math.max(this._state.batchingStats.maxBatchTokenCap ?? 0, maximum);
        if (bounded < executionPlan.batchSize) this._state.batchingStats.batchClampCount = (this._state.batchingStats.batchClampCount ?? 0) + 1;
        const batch = await prepareWholeTokenBatch(this._state, generatedIds, bounded);
        if (batch) {
          let stopped = false;
          const delivered = [];
          try {
            for await (const token of batch.tokens) {
              if (stopIfAborted()) { stopped = true; break; }
              batch.publish(); generatedIds.push(token); tokensGenerated++;
              const text = emitMode === 'token' ? '' : decodeToken(token);
              delivered.push({ id: token, text });
              yield emitMode === 'token' ? token : text;
              options.onToken?.(token, text);
              if (isStopToken(token, stopTokenIds, eosToken)) { this._recordStopReason('stop-token', token); stopped = true; break; }
              if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
                this._recordStopReason('stop-sequence', token); stopped = true; break;
              }
            }
            if (delivered.length) options.onBatch?.(delivered);
          } finally { await batch.settle(); }
          if (stopped) break;
          continue;
        }
      }

      if (useBatchPath) {
        const remaining = opts.maxTokens - tokensGenerated;
        const maxBatchDecodeTokens = batchTokenLimit(hotVocabularyBatchDecodeAvailable);
        const requestedBatchTokens = executionPlan.batchSize * intervalBatches;
        const boundedBatchTokens = maxBatchDecodeTokens == null
          ? requestedBatchTokens
          : Math.min(requestedBatchTokens, maxBatchDecodeTokens);
        const thisBatchSize = Math.min(boundedBatchTokens, remaining);
        this._state.batchingStats.requestedBatchTokens = Math.max(
          this._state.batchingStats.requestedBatchTokens ?? 0,
          requestedBatchTokens
        );
        this._state.batchingStats.effectiveBatchTokens = Math.max(
          this._state.batchingStats.effectiveBatchTokens ?? 0,
          thisBatchSize
        );
        if (maxBatchDecodeTokens != null) {
          this._state.batchingStats.maxBatchTokenCap = Math.max(
            this._state.batchingStats.maxBatchTokenCap ?? 0,
            maxBatchDecodeTokens
          );
        }
        if (thisBatchSize < requestedBatchTokens) {
          this._state.batchingStats.batchClampCount = (this._state.batchingStats.batchClampCount ?? 0) + 1;
        }
        const lastToken = generatedIds[generatedIds.length - 1];
        const boundedExecutionPlan = thisBatchSize < requestedBatchTokens
          ? {
            ...executionPlan,
            batchSize: Math.min(executionPlan.batchSize, thisBatchSize),
            readbackInterval: readbackInterval == null
              ? null
              : Math.min(intervalBatches, thisBatchSize),
          }
          : executionPlan;
        const batchOpts = boundedExecutionPlan === executionPlan
          ? opts
          : { ...opts, executionPlan: boundedExecutionPlan };

        let batchResult;
        try {
          batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, batchOpts);
        } catch (error) {
          if (stopIfAborted()) break;
          assertBatchRetryInPlaceAllowed(this._state, error);
          log.warn('Pipeline', `Batch decode failed, falling back to single-token: ${error}`);
          useBatchPath = false;
          let nextToken;
          try {
            nextToken = await decodeSingleTokenViaLogits();
          } catch (singleTokenError) {
            if (this._shouldUseFinitenessFallback(singleTokenError, `decode-batch-step-${tokensGenerated}`)) {
              log.warn('Pipeline', `FinitenessGuard caught NaN/Inf at batch step ${tokensGenerated}. Truncating KV cache and retrying token with F32 precision.`);
              nextToken = await this._retryDecodeStepWithFinitenessWindow(
                generatedIds,
                opts,
                `decode-batch-step-${tokensGenerated}`
              );
            } else {
              throw singleTokenError;
            }
          }
          if (stopIfAborted()) break;
          generatedIds.push(nextToken);
          tokensGenerated++;
          if (emitMode === 'token') {
            yield nextToken;
            if (options.onToken) options.onToken(nextToken, '');
          } else {
            const tokenText = decodeToken(nextToken);
            yield tokenText;
            if (options.onToken) options.onToken(nextToken, tokenText);
          }
          this._consumeFinitenessFallbackToken(opts);
          if (isStopToken(nextToken, stopTokenIds, eosToken)) {
            this._recordStopReason('stop-token', nextToken);
            break;
          }
          continue;
        }
        let batchTokens = [];
        let hitStop = false;
        let stopTokenId = null;
        for (const tokenId of batchResult.tokens) {
          if (options.signal?.aborted) break;
          generatedIds.push(tokenId);
          tokensGenerated++;
          if (emitMode === 'token') {
            yield tokenId;
            if (options.onToken) options.onToken(tokenId, '');
            batchTokens.push({ id: tokenId, text: '' });
          } else {
            const tokenText = decodeToken(tokenId);
            yield tokenText;
            if (options.onToken) options.onToken(tokenId, tokenText);
            batchTokens.push({ id: tokenId, text: tokenText });
          }
          if (batchTokens.length === executionPlan.batchSize) {
            if (options.onBatch) options.onBatch(batchTokens);
            batchTokens = [];
          }
          if (isStopToken(tokenId, stopTokenIds, eosToken)) {
            hitStop = true;
            stopTokenId = tokenId;
            break;
          }
        }
        if (batchTokens.length > 0 && options.onBatch) options.onBatch(batchTokens);
        if (stopIfAborted()) break;
        if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
          this._recordStopReason('stop-sequence', generatedIds[generatedIds.length - 1] ?? null);
          break;
        }
        if (hitStop) {
          this._recordStopReason('stop-token', stopTokenId);
          break;
        }
        if (shouldDisableBatchDecodeAfterShortBatch({
          hitStop,
          actualCount: batchResult.actualCount,
          requestedCount: thisBatchSize,
        })) {
          useBatchPath = false;
          continue;
        }
      } else if (opts.speculation?.mode === 'self') {
        // Self-speculation: decode one base token plus a configurable burst of
        // speculative tokens per iteration. Same-model speculation always
        // accepts under greedy because the model is deterministic — both base
        // and speculative use the same weights and state. The benefit is
        // amortizing per-iteration overhead for models where batch decode is
        // disabled (e.g., linear attention).
        const speculativeBurstTokens = opts.speculation.tokens;
        if (!Number.isInteger(speculativeBurstTokens) || speculativeBurstTokens < 1) {
          throw new Error('[Pipeline] resolved self-speculation tokens must be a positive integer.');
        }
        const doSpecDecode = hasLinearLayers
          ? () => this._decodeStep(generatedIds, opts)
          : decodeSingleTokenViaLogits;

        // Base decode
        let baseToken;
        try {
          baseToken = await doSpecDecode();
        } catch (error) {
          if (this._shouldUseFinitenessFallback(error, `spec-base-${tokensGenerated}`)) {
            log.warn('Pipeline', `FinitenessGuard caught NaN/Inf at step ${tokensGenerated} (speculation:base). Retrying.`);
            baseToken = await this._retryDecodeStepWithFinitenessWindow(generatedIds, opts, `spec-base-${tokensGenerated}`);
          } else {
            throw error;
          }
        }
        if (stopIfAborted()) break;
        generatedIds.push(baseToken);
        tokensGenerated++;
        if (emitMode === 'token') {
          yield baseToken;
          if (options.onToken) options.onToken(baseToken, '');
        } else {
          const text = decodeToken(baseToken);
          yield text;
          if (options.onToken) options.onToken(baseToken, text);
        }
        this._consumeFinitenessFallbackToken(opts);

        if (isStopToken(baseToken, stopTokenIds, eosToken)) {
          this._recordStopReason('stop-token', baseToken);
          break;
        }
        if (tokensGenerated >= opts.maxTokens) break;
        if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
          this._recordStopReason('stop-sequence', baseToken);
          break;
        }

        for (let specIndex = 0; specIndex < speculativeBurstTokens; specIndex += 1) {
          if (stopIfAborted()) break;
          if (tokensGenerated >= opts.maxTokens) {
            break;
          }
          let specToken;
          try {
            specToken = await doSpecDecode();
          } catch (error) {
            if (this._shouldUseFinitenessFallback(error, `spec-extra-${tokensGenerated}`)) {
              log.warn('Pipeline', `FinitenessGuard caught NaN/Inf at step ${tokensGenerated} (speculation:spec). Retrying.`);
              specToken = await this._retryDecodeStepWithFinitenessWindow(generatedIds, opts, `spec-extra-${tokensGenerated}`);
            } else {
              throw error;
            }
          }
          if (stopIfAborted()) break;
          generatedIds.push(specToken);
          tokensGenerated++;
          this._state.stats.speculationAttempts = (this._state.stats.speculationAttempts ?? 0) + 1;
          this._state.stats.speculationAccepted = (this._state.stats.speculationAccepted ?? 0) + 1;
          if (emitMode === 'token') {
            yield specToken;
            if (options.onToken) options.onToken(specToken, '');
          } else {
            const text = decodeToken(specToken);
            yield text;
            if (options.onToken) options.onToken(specToken, text);
          }
          this._consumeFinitenessFallbackToken(opts);

          if (opts.debug || opts.benchmark) {
            const elapsedMs = performance.now() - decodeStart;
            const tokPerSec = (Math.max(0, tokensGenerated - 1) / elapsedMs) * 1000;
            log.debug('Decode', `#${tokensGenerated} speculation:self (${tokPerSec.toFixed(2)} tok/s avg)`);
          }

          if (isStopToken(specToken, stopTokenIds, eosToken)) {
            this._recordStopReason('stop-token', specToken);
            break;
          }
          if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
            this._recordStopReason('stop-sequence', specToken);
            break;
          }
        }
        if (isStopToken(generatedIds[generatedIds.length - 1], stopTokenIds, eosToken)) {
          this._recordStopReason('stop-token', generatedIds[generatedIds.length - 1]);
          break;
        }
        if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
          this._recordStopReason('stop-sequence', generatedIds[generatedIds.length - 1] ?? null);
          break;
        }
      } else {
        const tokenStart = performance.now();
        let nextToken;
        try {
          nextToken = hasLinearLayers
            ? await this._decodeStep(generatedIds, opts)
            : await decodeSingleTokenViaLogits();
        } catch (error) {
          if (this._shouldUseFinitenessFallback(error, `decode-step-${tokensGenerated}`)) {
            log.warn('Pipeline', `FinitenessGuard caught NaN/Inf at step ${tokensGenerated}. Truncating KV cache and retrying token with F32 precision.`);
            nextToken = await this._retryDecodeStepWithFinitenessWindow(
              generatedIds,
              opts,
              `decode-step-${tokensGenerated}`
            );
          } else {
            throw error;
          }
        }
        const tokenTime = performance.now() - tokenStart;
        if (stopIfAborted()) break;
        generatedIds.push(nextToken);
        tokensGenerated++;

        // Step 5: Fire-and-forget prefetch of next token's PLE row.
        if (pleHiddenSize > 0) {
          const pleWeights = this._state.weights.get('per_layer_inputs');
          if (pleWeights?.embedTokensPerLayer) {
            this._state.plePrefetchPending = prefetchPerLayerRow(
              nextToken,
              pleWeights.embedTokensPerLayer,
              this._state.modelConfig.numLayers * pleHiddenSize,
              resolvedPerLayerInputsSession,
            );
          }
        }

        const tokenText = emitMode === 'token' ? '' : decodeToken(nextToken);
        if (emitMode === 'token') {
          yield nextToken;
          if (options.onToken) options.onToken(nextToken, '');
        } else {
          yield tokenText;
          if (options.onToken) options.onToken(nextToken, tokenText);
        }
        this._consumeFinitenessFallbackToken(opts);

        if (opts.debug || opts.benchmark) {
          const elapsedMs = performance.now() - decodeStart;
          const tokPerSec = (Math.max(0, tokensGenerated - 1) / elapsedMs) * 1000;
          log.debug('Decode', `#${tokensGenerated} "${tokenText}" ${tokenTime.toFixed(0)}ms (${tokPerSec.toFixed(2)} tok/s avg)`);
        }

        if (isStopToken(nextToken, stopTokenIds, eosToken)) {
          this._recordStopReason('stop-token', nextToken);
          break;
        }
        if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
          this._recordStopReason('stop-sequence', nextToken);
          break;
        }
      }
    }

    if (!this._state.stats.stopReason) {
      this._recordStopReason(tokensGenerated >= opts.maxTokens ? 'max-tokens' : 'completed');
    }
    this._state.stats.decodeTimeMs = performance.now() - decodeStart;
    this._state.stats.tokensGenerated = tokensGenerated;
    // Prefill produces the first token; decode timing covers only later tokens.
    this._state.stats.decodeTokens = Math.max(0, tokensGenerated - 1);
    this._state.stats.batching = { ...this._state.batchingStats };
  }

export async function _decodeStep(currentIds, opts) {
    if (usesReplayPrefillDecode(this._state)) {
      return this._decodeNextTokenViaLogits(currentIds, opts);
    }
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;
    const sampling = opts.temperature === 0 ? opts : {
      ...opts, seed: samplingSeedForToken(opts.seed, currentIds.length - opts.promptTokenCount),
    };
    return decodeStep(this._state, currentIds, sampling, this._getDecodeHelpers(debugCheckBuffer));
  }

export async function decodeStepLogits(currentIds, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    resetActiveExecutionPlan(this._state);

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    markDiagnosticSequenceStep(this._state, 'decode', options);
    return this._decodeStepToLogits(currentIds, opts);
  }

export async function advanceWithToken(tokenId, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    resetActiveExecutionPlan(this._state);
    assertIncrementalDecodeSupport(this._state, 'advanceWithToken');

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;

    this._assertTokenIdInRange(tokenId, 'advanceWithToken');
    await runAdvanceWithToken(this._state, tokenId, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

export async function advanceWithTokenAndEmbedding(tokenId, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    resetActiveExecutionPlan(this._state);
    assertIncrementalDecodeSupport(this._state, 'advanceWithTokenAndEmbedding');

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const embeddingMode = resolveAdvanceEmbeddingMode(this._state, options);
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;

    this._assertTokenIdInRange(tokenId, 'advanceWithTokenAndEmbedding');
    return runAdvanceWithTokenAndEmbedding(
      this._state,
      tokenId,
      opts,
      this._getDecodeHelpers(debugCheckBuffer),
      embeddingMode
    );
  }

export async function _generateNTokensGPU(startToken, N, currentIds, opts) {
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;
    return generateNTokensGPU(this._state, startToken, N, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }
