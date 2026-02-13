

import { getDevice, setTrackSubmits } from '../../gpu/device.js';
import { releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { isGPUSamplingAvailable } from '../../gpu/kernels/sample.js';
import { markWarmed as markKernelCacheWarmed } from '../../gpu/kernel-selection-cache.js';
import { resetSubmitStats, logSubmitStats } from '../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../gpu/command-recorder.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import { log, trace } from '../../debug/index.js';
import { validateCallTimeOptions } from '../../config/param-validator.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

// Pipeline sub-modules
import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from './sampling.js';
import { enforceLogitDrift } from '../../hotswap/intent-bundle.js';
import { applyChatTemplate, isStopToken } from './init.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { computeLogits, recordLogitsGPU, extractLastPositionLogits, applySoftcapping, rmsNormCPU } from './logits.js';
import { isWeightBuffer, isCpuWeightBuffer, getWeightDtype } from '../../gpu/weight-buffer.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
// Import as a namespace so the module can still link if the browser cache has a
// slightly older generator-steps.js (avoids hard "missing export" crashes).
import * as generatorSteps from './generator-steps.js';

const {
  decodeStep,
  decodeStepLogits,
  advanceWithToken,
  generateNTokensGPU,
  shouldUseBatchDecode,
  sumProfileTimings,
} = generatorSteps;

const advanceWithTokenAndEmbedding = generatorSteps.advanceWithTokenAndEmbedding ?? null;
import { buildLayerContext, debugCheckBuffer as debugCheckBufferHelper, getLogitsConfig, getLogitsWeights } from './generator-helpers.js';

import { decodeReadback, getLogitsHealth } from './debug-utils.js';

export class PipelineGenerator {
  
  #state;

  _assertTokenIdsInRange(tokenIds, context = 'encode') {
    const vocabSize = this.#state?.modelConfig?.vocabSize;
    if (!Array.isArray(tokenIds)) {
      throw new Error(`[Tokenizer] ${context}: expected tokenIds array, got ${typeof tokenIds}`);
    }
    if (!Number.isFinite(vocabSize) || vocabSize <= 0) {
      throw new Error(`[Tokenizer] ${context}: invalid model vocabSize=${vocabSize}`);
    }

    let firstBadIdx = -1;
    let firstBadId = -1;
    let maxId = -1;
    let badCount = 0;
    for (let i = 0; i < tokenIds.length; i++) {
      const id = tokenIds[i];
      if (!Number.isFinite(id) || id < 0 || id >= vocabSize) {
        badCount++;
        if (firstBadIdx < 0) {
          firstBadIdx = i;
          firstBadId = id;
        }
      }
      if (Number.isFinite(id) && id > maxId) maxId = id;
    }
    if (badCount === 0) return;

    const tok = this.#state?.tokenizer;
    const tokenizerVocabSize = tok?.getVocabSize?.() ?? null;
    let badText = null;
    try {
      badText = tok?.decode?.([firstBadId], false, false) ?? null;
    } catch {
      badText = null;
    }

    throw new Error(
      `[Tokenizer] ${context}: token id out of range for model vocab. ` +
      `modelVocabSize=${vocabSize}, tokenizerVocabSize=${tokenizerVocabSize ?? 'unknown'}, ` +
      `badCount=${badCount}/${tokenIds.length}, firstBadIdx=${firstBadIdx}, firstBadId=${firstBadId}` +
      (badText ? ` ("${badText}")` : '') +
      `, maxId=${maxId}. ` +
      'This will poison GPU embedding gather (NaNs). Fix by re-converting the model or aligning tokenizer.json IDs to embedding/LM-head shapes.'
    );
  }

  _assertTokenIdInRange(tokenId, context = 'token') {
    const vocabSize = this.#state?.modelConfig?.vocabSize;
    if (!Number.isFinite(vocabSize) || vocabSize <= 0) {
      throw new Error(`[Tokenizer] ${context}: invalid model vocabSize=${vocabSize}`);
    }
    if (!Number.isFinite(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
      const tok = this.#state?.tokenizer;
      const tokenizerVocabSize = tok?.getVocabSize?.() ?? null;
      throw new Error(
        `[Tokenizer] ${context}: tokenId=${tokenId} out of range (modelVocabSize=${vocabSize}, tokenizerVocabSize=${tokenizerVocabSize ?? 'unknown'}).`
      );
    }
  }

  
  constructor(state) {
    this.#state = state;
  }

  _resolveStepOptions(options = {}) {
    const runtimeDefaults = this.#state.runtimeConfig.inference;
    const samplingDefaults = runtimeDefaults.sampling;
    const batchingDefaults = runtimeDefaults.batching;
    const generationDefaults = runtimeDefaults.generation;

    return {
      temperature: options.temperature ?? samplingDefaults.temperature,
      topP: options.topP ?? samplingDefaults.topP,
      topK: options.topK ?? samplingDefaults.topK,
      repetitionPenalty: options.repetitionPenalty ?? samplingDefaults.repetitionPenalty,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? generationDefaults.profile,
      disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
      batchSize: options.batchSize ?? batchingDefaults.batchSize,
      stopCheckMode: options.stopCheckMode ?? batchingDefaults.stopCheckMode,
    };
  }

  _getDecodeHelpers(debugCheckBuffer) {
    return {
      buildLayerContext: (recorder, isDecodeMode, debugLayers) =>
        buildLayerContext(this.#state, recorder, isDecodeMode, debugLayers, debugCheckBuffer),
      getLogitsWeights: () => getLogitsWeights(this.#state),
      getLogitsConfig: () => getLogitsConfig(this.#state),
      debugCheckBuffer,
    };
  }

  _resolveFloatDtypeFromByteSize(totalBytes, expectedLength, fallback = 'f32') {
    if (!Number.isFinite(totalBytes) || totalBytes <= 0 || !Number.isFinite(expectedLength) || expectedLength <= 0) {
      return fallback;
    }
    const bytesPerElement = totalBytes / expectedLength;
    if (Math.abs(bytesPerElement - 2) < 0.5) return 'f16';
    if (Math.abs(bytesPerElement - 4) < 0.5) return 'f32';
    return bytesPerElement < 3 ? 'f16' : 'f32';
  }

  _decodeFloatWeights(data, dtype, expectedLength, label) {
    const decodeDtype = dtype === 'bf16'
      ? 'bf16'
      : (dtype === 'f16' ? 'f16' : 'f32');
    const decoded = decodeReadback(data, decodeDtype);
    if (decoded.length !== expectedLength) {
      throw new Error(
        `[Pipeline] ${label} length mismatch: expected=${expectedLength}, got=${decoded.length}`
      );
    }
    return decoded;
  }

  async _getFinalNormWeights() {
    const hiddenSize = this.#state.modelConfig.hiddenSize;
    const finalNorm = this.#state.weights.get('final_norm');
    if (!finalNorm) {
      throw new Error('[Pipeline] final_norm weight is missing; cannot extract embedding.');
    }

    let weights;

    if (finalNorm instanceof Float32Array) {
      weights = finalNorm;
    } else if (isCpuWeightBuffer(finalNorm)) {
      const dtype = finalNorm.dtype === 'bf16' ? 'bf16' : (finalNorm.dtype === 'f16' ? 'f16' : 'f32');
      const data = finalNorm.data;
      if (!(data instanceof Float32Array) && !ArrayBuffer.isView(data)) {
        throw new Error('[Pipeline] final_norm CPU weight buffer has unsupported data type.');
      }
      const bytes = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      weights = this._decodeFloatWeights(bytes, dtype, hiddenSize, 'final_norm');
    } else if (isWeightBuffer(finalNorm)) {
      const dtype = finalNorm.dtype === 'bf16' ? 'bf16' : (finalNorm.dtype === 'f16' ? 'f16' : 'f32');
      const bytesPerElement = dtype === 'f16' || dtype === 'bf16' ? 2 : 4;
      const readSize = hiddenSize * bytesPerElement;
      const data = await readBuffer(finalNorm.buffer, readSize);
      if (data.byteLength === 0) {
        throw new Error('[Pipeline] final_norm readback returned empty buffer.');
      }
      weights = this._decodeFloatWeights(data, dtype, hiddenSize, 'final_norm');
    } else if (finalNorm instanceof GPUBuffer) {
      const dtype = this._resolveFloatDtypeFromByteSize(finalNorm.size, hiddenSize, 'f32');
      const bytesPerElement = dtype === 'f16' ? 2 : 4;
      const readSize = hiddenSize * bytesPerElement;
      const data = await readBuffer(finalNorm, readSize);
      if (data.byteLength === 0) {
        throw new Error('[Pipeline] final_norm readback returned empty buffer.');
      }
      weights = this._decodeFloatWeights(data, dtype, hiddenSize, 'final_norm');
    } else if (ArrayBuffer.isView(finalNorm)) {
      const view = finalNorm;
      const dtype = this._resolveFloatDtypeFromByteSize(view.byteLength, hiddenSize, 'f32');
      const bytes = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
      weights = this._decodeFloatWeights(bytes, dtype, hiddenSize, 'final_norm');
    } else {
      throw new Error('[Pipeline] final_norm weight has unsupported type.');
    }
    if (!(weights instanceof Float32Array) || weights.length !== hiddenSize) {
      throw new Error(
        `[Pipeline] final_norm length mismatch: expected=${hiddenSize}, got=${weights?.length ?? 'unknown'}`
      );
    }
    return weights;
  }

  _extractEmbeddingFromHidden(hiddenStates, numTokens, hiddenSize, embeddingMode, finalNormWeights, config) {
    const expectedLength = numTokens * hiddenSize;
    if (hiddenStates.length !== expectedLength) {
      throw new Error(
        `[Pipeline] Hidden state length mismatch for embedding extraction: expected=${expectedLength}, got=${hiddenStates.length}`
      );
    }

    const applyFinalNorm = (tokenIndex) => {
      const offset = tokenIndex * hiddenSize;
      const tokenHidden = hiddenStates.subarray(offset, offset + hiddenSize);
      return rmsNormCPU(
        tokenHidden,
        finalNormWeights,
        config.rmsNormEps,
        config.rmsNormWeightOffset
      );
    };

    if (embeddingMode === 'last') {
      return applyFinalNorm(numTokens - 1);
    }

    if (embeddingMode === 'mean') {
      const pooled = new Float32Array(hiddenSize);
      for (let t = 0; t < numTokens; t++) {
        const tokenEmbedding = applyFinalNorm(t);
        for (let i = 0; i < hiddenSize; i++) {
          pooled[i] += tokenEmbedding[i];
        }
      }
      const invTokens = numTokens > 0 ? (1 / numTokens) : 1;
      for (let i = 0; i < hiddenSize; i++) {
        pooled[i] *= invTokens;
      }
      return pooled;
    }

    throw new Error(`prefillWithEmbedding: unsupported embeddingMode "${embeddingMode}" (expected "last" or "mean")`);
  }

  // ==========================================================================
  // Generation Public API
  // ==========================================================================

  
  async *generate(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);

    this.#state.isGenerating = true;
    this.#state.decodeStepCount = 0;
    this.#state.disableRecordedLogits = false;
    this.#state.disableFusedDecode = false;
    this.#state.decodeRing?.reset();
    this.#state.stats.gpuTimePrefillMs = undefined;
    this.#state.stats.gpuTimeDecodeMs = undefined;
    this.#state.stats.decodeRecordMs = 0;
    this.#state.stats.decodeSubmitWaitMs = 0;
    this.#state.stats.decodeReadbackWaitMs = 0;
    this.#state.stats.ttftMs = 0;
    const startTime = performance.now();

    const runtimeDefaults = this.#state.runtimeConfig.inference;
    const samplingDefaults = runtimeDefaults.sampling;
    const batchingDefaults = runtimeDefaults.batching;
    const generationDefaults = runtimeDefaults.generation;

    const opts = {
      maxTokens: options.maxTokens ?? batchingDefaults.maxTokens,
      temperature: options.temperature ?? samplingDefaults.temperature,
      topP: options.topP ?? samplingDefaults.topP,
      topK: options.topK ?? samplingDefaults.topK,
      repetitionPenalty: options.repetitionPenalty ?? samplingDefaults.repetitionPenalty,
      stopSequences: options.stopSequences ?? [],
      useSpeculative: options.useSpeculative ?? generationDefaults.useSpeculative,
      useChatTemplate: options.useChatTemplate
        ?? this.#state.runtimeConfig.inference.chatTemplate?.enabled
        ?? this.#state.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? generationDefaults.profile,
      benchmark: options.benchmark ?? generationDefaults.benchmark,
      disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
      batchSize: options.batchSize ?? batchingDefaults.batchSize,
      stopCheckMode: options.stopCheckMode ?? batchingDefaults.stopCheckMode,
    };

    if (opts.debug) {
      log.debug('Pipeline', `ChatTemplate: options=${options.useChatTemplate}, final=${opts.useChatTemplate}`);
    }

    try {
      let processedPrompt = prompt;
      if (opts.useChatTemplate && this.#state.modelConfig.chatTemplateType) {
        processedPrompt = applyChatTemplate(prompt, this.#state.modelConfig.chatTemplateType);
        if (opts.debug) log.debug('Pipeline', `Applied ${this.#state.modelConfig.chatTemplateType} chat template`);
      }

      const inputIds = this.#state.tokenizer.encode(processedPrompt);
      this._assertTokenIdsInRange(inputIds, 'generate.encode');
      const generatedIds = [...inputIds];
      this.#state.stats.prefillTokens = inputIds.length;

      if (opts.debug) {
        log.debug('Pipeline', `Input: ${inputIds.length} tokens`);
      }

      const prefillStart = performance.now();
      const prefillLogits = await this._prefill(inputIds, opts);
      this.#state.stats.prefillTimeMs = performance.now() - prefillStart;

      const intentBundleConfig = this.#state.runtimeConfig.shared.intentBundle;
      const intentBundle = intentBundleConfig?.bundle;
      const expectedTopK = intentBundle?.payload?.expectedTopK
        ?? intentBundle?.payload?.expected_top_k;
      const maxDriftThreshold = intentBundle?.constraints?.maxDriftThreshold
        ?? intentBundle?.constraints?.max_drift_threshold;

      if (intentBundleConfig?.enabled && Array.isArray(expectedTopK) && expectedTopK.length > 0) {
        const actualTopK = getTopK(
          prefillLogits,
          expectedTopK.length,
          (tokens) => this.#state.tokenizer?.decode?.(tokens) || '?'
        ).map((token) => token.token);
        const driftResult = enforceLogitDrift(expectedTopK, actualTopK, maxDriftThreshold);
        if (!driftResult.ok) {
          throw new Error(`Intent bundle drift check failed: ${driftResult.reason}`);
        }
      }

      applyRepetitionPenalty(prefillLogits, generatedIds, opts.repetitionPenalty);
      const padTokenId = this.#state.tokenizer?.getSpecialTokens?.()?.pad;

      if (opts.debug) {
        const topAfterPenalty = getTopK(prefillLogits, 5, (tokens) => this.#state.tokenizer?.decode?.(tokens) || '?');
        log.debug('Pipeline', `After rep penalty top-5: ${topAfterPenalty.map(t => `"${t.text}"(${(t.prob * 100).toFixed(1)}%)`).join(', ')}`);
      }

      const firstToken = sample(prefillLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
        padTokenId,
      });

      if (opts.debug) {
        log.debug('Pipeline', `First token sampled: id=${firstToken} text="${this.#state.tokenizer?.decode?.([firstToken]) || '?'}"`);
      }

      generatedIds.push(firstToken);
      this.#state.stats.ttftMs = performance.now() - startTime;

      const decodeToken = (tokenId) => {
        const text = this.#state.tokenizer.decode([tokenId], true, false);
        if (text.length > 0) return text;
        const raw = this.#state.tokenizer.decode([tokenId], false, false);
        if (raw.length > 0) return raw;
        return `[${tokenId}]`;
      };

      const firstText = decodeToken(firstToken);
      yield firstText;
      if (options.onToken) options.onToken(firstToken, firstText);

      const stopTokenIds = this.#state.modelConfig.stopTokenIds;
      const eosToken = this.#state.tokenizer.getSpecialTokens?.()?.eos;
      let tokensGenerated = 1;

      markKernelCacheWarmed();

      const decodeStart = performance.now();
      const lmHead = this.#state.weights.get('lm_head');
      const embedBuffer = this.#state.weights.get('embed');
      const hasCpuWeights = isCpuWeightBuffer(lmHead)
        || isCpuWeightBuffer(embedBuffer)
        || lmHead instanceof Float32Array
        || embedBuffer instanceof Float32Array;
      const gpuSamplingAvailable = isGPUSamplingAvailable() && !hasCpuWeights;
      let useBatchPath = shouldUseBatchDecode({
        batchSize: opts.batchSize,
        useGPU: this.#state.useGPU,
        gpuSamplingAvailable,
        disableMultiTokenDecode: opts.disableMultiTokenDecode,
        disableCommandBatching: opts.disableCommandBatching,
      });
      const intervalBatches = batchingDefaults.readbackInterval == null
        ? 1
        : batchingDefaults.readbackInterval;

      if (opts.debug && useBatchPath) {
        log.debug(
          'Pipeline',
          `Using batch decode path with batchSize=${opts.batchSize}, stopCheckMode=${opts.stopCheckMode}, readbackInterval=${batchingDefaults.readbackInterval}`
        );
      }

      while (tokensGenerated < opts.maxTokens) {
        if (options.signal?.aborted) break;

        if (useBatchPath) {
          const remaining = opts.maxTokens - tokensGenerated;
          const thisBatchSize = Math.min(opts.batchSize * intervalBatches, remaining);
          const lastToken = generatedIds[generatedIds.length - 1];

          try {
            const batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, opts);

            
            let batchTokens = [];
            for (const tokenId of batchResult.tokens) {
              generatedIds.push(tokenId);
              tokensGenerated++;

              const tokenText = decodeToken(tokenId);
              yield tokenText;
              if (options.onToken) options.onToken(tokenId, tokenText);
              batchTokens.push({ id: tokenId, text: tokenText });
              if (batchTokens.length === opts.batchSize) {
                if (options.onBatch) options.onBatch(batchTokens);
                batchTokens = [];
              }
            }

            if (batchTokens.length > 0 && options.onBatch) options.onBatch(batchTokens);

            if (batchResult.actualCount < thisBatchSize) {
              break;
            }

            if (opts.stopSequences.length > 0) {
              const fullText = this.#state.tokenizer.decode(generatedIds.slice(inputIds.length), false);
              if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
            }
          } catch (error) {
            log.warn('Pipeline', `Batch decode failed, falling back to single-token: ${error}`);
            useBatchPath = false;
            const nextToken = await this._decodeStep(generatedIds, opts);
            generatedIds.push(nextToken);
            tokensGenerated++;

            const tokenText = decodeToken(nextToken);
            yield tokenText;
            if (options.onToken) options.onToken(nextToken, tokenText);

            if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
          }
        } else {
          const tokenStart = performance.now();
          const nextToken = await this._decodeStep(generatedIds, opts);
          const tokenTime = performance.now() - tokenStart;
          generatedIds.push(nextToken);
          tokensGenerated++;

          const tokenText = decodeToken(nextToken);
          yield tokenText;
          if (options.onToken) options.onToken(nextToken, tokenText);

          if (opts.debug || opts.benchmark) {
            const elapsedMs = performance.now() - decodeStart;
            const tokPerSec = (tokensGenerated / elapsedMs) * 1000;
            log.debug('Decode', `#${tokensGenerated} "${tokenText}" ${tokenTime.toFixed(0)}ms (${tokPerSec.toFixed(2)} tok/s avg)`);
          }

          if (isStopToken(nextToken, stopTokenIds, eosToken)) break;

          if (opts.stopSequences.length > 0) {
            const fullText = this.#state.tokenizer.decode(generatedIds.slice(inputIds.length), false);
            if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
          }
        }
      }

      this.#state.stats.decodeTimeMs = performance.now() - decodeStart;
      this.#state.stats.tokensGenerated = tokensGenerated;
      this.#state.stats.decodeTokens = tokensGenerated;
      this.#state.stats.totalTimeMs = performance.now() - startTime;

      if (opts.debug) {
        log.debug('Pipeline', `Generated ${tokensGenerated} tokens in ${this.#state.stats.totalTimeMs.toFixed(0)}ms`);
      }

      const ttft = this.#state.stats.ttftMs || this.#state.stats.prefillTimeMs;
      const decodeTokens = Math.max(0, tokensGenerated - 1);
      const decodeSpeed = decodeTokens > 0 ? (decodeTokens / this.#state.stats.decodeTimeMs * 1000) : 0;
      if (opts.benchmark) {
        log.info('Benchmark', `TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this.#state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this.#state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      } else {
        log.info('Perf', `TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this.#state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this.#state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      }
      trace.perf('Decode summary', {
        ttftMs: ttft,
        prefillMs: this.#state.stats.prefillTimeMs,
        decodeMs: this.#state.stats.decodeTimeMs,
        decodeTokens,
        decodeSpeed,
        totalMs: this.#state.stats.totalTimeMs,
      });
    } finally {
      this.#state.isGenerating = false;
    }
  }

  
  async prefillKVOnly(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    this.#state.stats.gpuTimePrefillMs = undefined;

    const generationDefaults = this.#state.runtimeConfig.inference.generation;

    const opts = {
      useChatTemplate: options.useChatTemplate
        ?? this.#state.runtimeConfig.inference.chatTemplate?.enabled
        ?? this.#state.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? generationDefaults.profile,
      disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
    };

    let processedPrompt = prompt;
    if (opts.useChatTemplate && this.#state.modelConfig.chatTemplateType) {
      processedPrompt = applyChatTemplate(prompt, this.#state.modelConfig.chatTemplateType);
    }

    const inputIds = this.#state.tokenizer.encode(processedPrompt);
    this._assertTokenIdsInRange(inputIds, 'prefillKVOnly.encode');
    if (opts.debug) {
      log.debug('Pipeline', `PrefillKVOnly: ${inputIds.length} tokens`);
    }

    const {
      numTokens,
      startPos,
      currentRecorder,
      recordProfile,
      currentHiddenBuffer,
    } = await this._prefillToHidden(inputIds, opts);

    // Ensure prefill work completes before returning a usable snapshot.
    if (currentRecorder) {
      await currentRecorder.submitAndWait();
      await recordProfile(currentRecorder);
    } else {
      const device = getDevice();
      if (device) {
        await device.queue.onSubmittedWorkDone();
      }
    }

    this.#state.currentSeqLen = startPos + numTokens;
    releaseBuffer(currentHiddenBuffer);

    const snapshot = this.#state.kvCache?.clone();
    if (!snapshot) {
      throw new Error('KV cache unavailable after prefill');
    }

    return {
      cache: snapshot,
      seqLen: this.#state.currentSeqLen,
      tokens: inputIds,
    };
  }

  async prefillWithEmbedding(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    this.#state.stats.gpuTimePrefillMs = undefined;

    const generationDefaults = this.#state.runtimeConfig.inference.generation;

    const modelType = String(this.#state.manifest?.modelType || '').toLowerCase();
    const defaultEmbeddingMode = modelType === 'embedding'
      ? 'mean'
      : generationDefaults.embeddingMode;

    const opts = {
      useChatTemplate: options.useChatTemplate
        ?? this.#state.runtimeConfig.inference.chatTemplate?.enabled
        ?? this.#state.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? generationDefaults.profile,
      disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
      embeddingMode: options.embeddingMode ?? defaultEmbeddingMode,
    };

    let processedPrompt = prompt;
    if (opts.useChatTemplate && this.#state.modelConfig.chatTemplateType) {
      processedPrompt = applyChatTemplate(prompt, this.#state.modelConfig.chatTemplateType);
    }

    const inputIds = this.#state.tokenizer.encode(processedPrompt);
    this._assertTokenIdsInRange(inputIds, 'prefillWithEmbedding.encode');
    if (opts.debug) {
      log.debug('Pipeline', `PrefillWithEmbedding: ${inputIds.length} tokens (mode=${opts.embeddingMode})`);
    }

    const {
      numTokens,
      config,
      startPos,
      activationDtype,
      activationBytes,
      currentRecorder,
      recordProfile,
      currentHiddenBuffer,
    } = await this._prefillToHidden(inputIds, opts);

    // Ensure prefill work completes before readback.
    if (currentRecorder) {
      await currentRecorder.submitAndWait();
      await recordProfile(currentRecorder);
    } else {
      const device = getDevice();
      if (device) {
        await device.queue.onSubmittedWorkDone();
      }
    }

    if (!allowReadback('pipeline.prefill.embedding')) {
      throw new Error('GPU readback disabled; cannot return embedding');
    }

    let embedding;
    try {
      const hiddenSize = config.hiddenSize;
      const hiddenBytes = numTokens * hiddenSize * activationBytes;
      const hiddenData = await readBuffer(currentHiddenBuffer, hiddenBytes);
      if (hiddenData.byteLength === 0) {
        throw new Error('GPU readback disabled; cannot return embedding');
      }
      const hiddenStates = decodeReadback(hiddenData, activationDtype);
      const finalNormWeights = await this._getFinalNormWeights();
      embedding = this._extractEmbeddingFromHidden(
        hiddenStates,
        numTokens,
        hiddenSize,
        opts.embeddingMode,
        finalNormWeights,
        config
      );
    } finally {
      releaseBuffer(currentHiddenBuffer);
    }

    this.#state.currentSeqLen = startPos + numTokens;

    const snapshot = this.#state.kvCache?.clone();
    if (!snapshot) {
      throw new Error('KV cache unavailable after prefill');
    }

    return {
      cache: snapshot,
      seqLen: this.#state.currentSeqLen,
      tokens: inputIds,
      embedding,
      embeddingMode: opts.embeddingMode,
    };
  }

  async prefillWithLogits(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    this.#state.stats.gpuTimePrefillMs = undefined;

    const generationDefaults = this.#state.runtimeConfig.inference.generation;

    const opts = {
      useChatTemplate: options.useChatTemplate
        ?? this.#state.runtimeConfig.inference.chatTemplate?.enabled
        ?? this.#state.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? generationDefaults.profile,
      disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
    };

    let processedPrompt = prompt;
    if (opts.useChatTemplate && this.#state.modelConfig.chatTemplateType) {
      processedPrompt = applyChatTemplate(prompt, this.#state.modelConfig.chatTemplateType);
    }

    const inputIds = this.#state.tokenizer.encode(processedPrompt);
    this._assertTokenIdsInRange(inputIds, 'prefillWithLogits.encode');
    if (opts.debug) {
      log.debug('Pipeline', `PrefillWithLogits: ${inputIds.length} tokens`);
    }

    const logits = await this._prefill(inputIds, opts);

    const snapshot = this.#state.kvCache?.clone();
    if (!snapshot) {
      throw new Error('KV cache unavailable after prefill');
    }

    return {
      cache: snapshot,
      seqLen: this.#state.currentSeqLen,
      tokens: inputIds,
      logits,
    };
  }

  
  async *generateWithPrefixKV(prefix, prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);

    // Apply snapshot
    this.#state.kvCache = prefix.cache.clone();
    if (this.#state.useGPU && this.#state.kvCache) {
      const device = getDevice();
      if (device) {
        this.#state.kvCache.setGPUContext({ device });
      }
    }
    this.#state.currentSeqLen = prefix.seqLen;

    this.#state.isGenerating = true;
    this.#state.decodeStepCount = 0;
    this.#state.stats.gpuTimePrefillMs = undefined;
    this.#state.stats.gpuTimeDecodeMs = undefined;
    this.#state.decodeRing?.reset();
    this.#state.stats.decodeRecordMs = 0;
    this.#state.stats.decodeSubmitWaitMs = 0;
    this.#state.stats.decodeReadbackWaitMs = 0;
    this.#state.stats.ttftMs = 0;
    const startTime = performance.now();

    const runtimeDefaults = this.#state.runtimeConfig.inference;
    const samplingDefaults = runtimeDefaults.sampling;
    const batchingDefaults = runtimeDefaults.batching;
    const generationDefaults = runtimeDefaults.generation;

    const opts = {
      maxTokens: options.maxTokens ?? batchingDefaults.maxTokens,
      temperature: options.temperature ?? samplingDefaults.temperature,
      topP: options.topP ?? samplingDefaults.topP,
      topK: options.topK ?? samplingDefaults.topK,
      repetitionPenalty: options.repetitionPenalty ?? samplingDefaults.repetitionPenalty,
      stopSequences: options.stopSequences ?? [],
      useSpeculative: options.useSpeculative ?? generationDefaults.useSpeculative,
      useChatTemplate: options.useChatTemplate
        ?? this.#state.runtimeConfig.inference.chatTemplate?.enabled
        ?? this.#state.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? generationDefaults.profile,
      benchmark: options.benchmark ?? generationDefaults.benchmark,
      disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
      batchSize: options.batchSize ?? batchingDefaults.batchSize,
      stopCheckMode: options.stopCheckMode ?? batchingDefaults.stopCheckMode,
    };

    try {
      let processedPrompt = prompt;
      if (opts.useChatTemplate && this.#state.modelConfig.chatTemplateType) {
        processedPrompt = applyChatTemplate(prompt, this.#state.modelConfig.chatTemplateType);
      }

      const inputIds = this.#state.tokenizer.encode(processedPrompt);
      this._assertTokenIdsInRange(inputIds, 'generateWithPrefixKV.encode');
      const generatedIds = [...prefix.tokens, ...inputIds];
      const promptTokenCount = generatedIds.length;
      this.#state.stats.prefillTokens = inputIds.length;

      const prefillStart = performance.now();
      const prefillLogits = await this._prefill(inputIds, opts);
      this.#state.stats.prefillTimeMs = performance.now() - prefillStart;

      applyRepetitionPenalty(prefillLogits, generatedIds, opts.repetitionPenalty);
      const padTokenId = this.#state.tokenizer?.getSpecialTokens?.()?.pad;
      const firstToken = sample(prefillLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
        padTokenId,
      });

      generatedIds.push(firstToken);
      this.#state.stats.ttftMs = performance.now() - startTime;

      const firstText = this.#state.tokenizer.decode([firstToken], true, false);
      yield firstText;
      if (options.onToken) options.onToken(firstToken, firstText);

      const stopTokenIds = this.#state.modelConfig.stopTokenIds;
      const eosToken = this.#state.tokenizer.getSpecialTokens?.()?.eos;
      let tokensGenerated = 1;

      markKernelCacheWarmed();

      const decodeStart = performance.now();
      const lmHead = this.#state.weights.get('lm_head');
      const embedBuffer = this.#state.weights.get('embed');
      const hasCpuWeights = isCpuWeightBuffer(lmHead)
        || isCpuWeightBuffer(embedBuffer)
        || lmHead instanceof Float32Array
        || embedBuffer instanceof Float32Array;
      const gpuSamplingAvailable = isGPUSamplingAvailable() && !hasCpuWeights;
      let useBatchPath = shouldUseBatchDecode({
        batchSize: opts.batchSize,
        useGPU: this.#state.useGPU,
        gpuSamplingAvailable,
        disableMultiTokenDecode: opts.disableMultiTokenDecode,
        disableCommandBatching: opts.disableCommandBatching,
      });
      const intervalBatches = batchingDefaults.readbackInterval == null
        ? 1
        : batchingDefaults.readbackInterval;

      while (tokensGenerated < opts.maxTokens) {
        if (options.signal?.aborted) break;

        if (useBatchPath) {
          const remaining = opts.maxTokens - tokensGenerated;
          const thisBatchSize = Math.min(opts.batchSize * intervalBatches, remaining);
          const lastToken = generatedIds[generatedIds.length - 1];

          try {
            const batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, opts);
            
            let batchTokens = [];
            for (const tokenId of batchResult.tokens) {
              generatedIds.push(tokenId);
              tokensGenerated++;
              const tokenText = this.#state.tokenizer.decode([tokenId], true, false);
              yield tokenText;
              if (options.onToken) options.onToken(tokenId, tokenText);
              batchTokens.push({ id: tokenId, text: tokenText });
              if (batchTokens.length === opts.batchSize) {
                if (options.onBatch) options.onBatch(batchTokens);
                batchTokens = [];
              }
            }
            if (batchTokens.length > 0 && options.onBatch) options.onBatch(batchTokens);
            if (batchResult.actualCount < thisBatchSize) break;
            if (opts.stopSequences.length > 0) {
              const fullText = this.#state.tokenizer.decode(generatedIds.slice(promptTokenCount), false);
              if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
            }
          } catch (error) {
            log.warn('Pipeline', `Batch decode failed, falling back to single-token: ${error}`);
            useBatchPath = false;
            const nextToken = await this._decodeStep(generatedIds, opts);
            generatedIds.push(nextToken);
            tokensGenerated++;
            const tokenText = this.#state.tokenizer.decode([nextToken], true, false);
            yield tokenText;
            if (options.onToken) options.onToken(nextToken, tokenText);
            if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
          }
        } else {
          const tokenStart = performance.now();
          const nextToken = await this._decodeStep(generatedIds, opts);
          const tokenTime = performance.now() - tokenStart;
          generatedIds.push(nextToken);
          tokensGenerated++;
          const tokenText = this.#state.tokenizer.decode([nextToken], true, false);
          yield tokenText;
          if (options.onToken) options.onToken(nextToken, tokenText);

          if (opts.debug || opts.benchmark) {
            const elapsedMs = performance.now() - decodeStart;
            const tokPerSec = (tokensGenerated / elapsedMs) * 1000;
            log.debug('Decode', `#${tokensGenerated} "${tokenText}" ${tokenTime.toFixed(0)}ms (${tokPerSec.toFixed(2)} tok/s avg)`);
          }

          if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
          if (opts.stopSequences.length > 0) {
            const fullText = this.#state.tokenizer.decode(generatedIds.slice(promptTokenCount), false);
            if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
          }
        }
      }

      this.#state.stats.decodeTimeMs = performance.now() - decodeStart;
      this.#state.stats.tokensGenerated = tokensGenerated;
      this.#state.stats.decodeTokens = tokensGenerated;
      this.#state.stats.totalTimeMs = performance.now() - startTime;
    } finally {
      this.#state.isGenerating = false;
    }
  }

  // ==========================================================================
  // Internal Methods (Prefill, Decode, Helpers)
  // ==========================================================================

  async _prefillToHidden(inputIds, opts) {
    // Internal-only: reuse the main prefill implementation but stop before logits.
    return this._prefill(inputIds, { ...opts, _returnHidden: true });
  }

  
  async _prefill(inputIds, opts) {
    const numTokens = inputIds.length;
    const config = this.#state.modelConfig;
    const startPos = this.#state.currentSeqLen;
    const returnHidden = opts?._returnHidden === true;
    this.#state.stats.gpuTimePrefillMs = undefined;

    const embedBufferRaw = this.#state.weights.get('embed');
    if (!(embedBufferRaw instanceof GPUBuffer) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
      throw new Error('Embed buffer not found or not a supported buffer type');
    }
    const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
    const embedDtype = isWeightBuffer(embedBufferRaw)
      ? getWeightDtype(embedBufferRaw)
      : isCpuWeightBuffer(embedBufferRaw)
        ? embedBufferRaw.dtype
        : null;
    if (opts.debug) {
      const embedSize = embedBuffer instanceof GPUBuffer ? embedBuffer.size : 'N/A';
      log.debug('Pipeline', `Embed buffer: type=${embedBuffer?.constructor?.name}, size=${embedSize}, dtype=${embedDtype}`);
    }

    const device = getDevice();
    const useCheckpoints = opts.debugLayers && opts.debugLayers.length > 0;
    const disableCommandBatching = opts.disableCommandBatching === true || opts.debug === true;
    const createRecorder = (label) => {
      if (!device || disableCommandBatching) return undefined;
      return opts.profile ? createProfilingRecorder(label) : createCommandRecorder(label);
    };
    const recorder = createRecorder('prefill');
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;
    const context = buildLayerContext(this.#state, recorder, false, opts.debugLayers, debugCheckBuffer);
    let gpuTimePrefillMs = 0;
    let hasGpuTimePrefill = false;
    const recordProfile = async (rec) => {
      if (!opts.profile || !rec?.isProfilingEnabled()) return;
      const timings = await rec.resolveProfileTimings();
      const total = sumProfileTimings(timings);
      if (total !== null) {
        gpuTimePrefillMs += total;
        hasGpuTimePrefill = true;
      }
      if (timings) {
        log.warn('Profile', `Prefill (${rec.label}):`);
        log.warn('Profile', CommandRecorder.formatProfileReport(timings));
      }
    };

    const benchmarkSubmits = opts.debug;
    if (benchmarkSubmits) {
      setTrackSubmits(true);
      resetSubmitStats();
    }

    const activationDtype = this.#state.runtimeConfig.inference.compute.activationDtype;
    const activationBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
    let hiddenStates = await embed(inputIds, embedBuffer, {
      hiddenSize: config.hiddenSize,
      vocabSize: config.vocabSize,
      scaleEmbeddings: config.scaleEmbeddings,
      debug: opts.debug,
      recorder,
      transpose: this.#state.embeddingTranspose,
      debugProbes: this.#state.runtimeConfig.shared.debug.probes,
      activationDtype,
      embeddingDtype: selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype: embedDtype }),
    });

    if (opts.debug && hiddenStates instanceof GPUBuffer) {
      if (recorder) {
        await recorder.submitAndWait();
        await recordProfile(recorder);
      }
      const debugReadbackSize = this.#state.runtimeConfig.shared.debug.pipeline.readbackSampleSize;
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
    let currentRecorder = recorder;
    
    let currentHiddenBuffer = hiddenStates.buffer;
    for (let l = 0; l < config.numLayers; l++) {
      context.recorder = currentRecorder;

      const prevBuffer = currentHiddenBuffer;
      const layerOutput = await processLayer(l, currentHiddenBuffer, numTokens, true, context);
      if (!(layerOutput instanceof GPUBuffer)) throw new Error('Expected GPUBuffer from processLayer');
      currentHiddenBuffer = layerOutput;

      const isCheckpoint = useCheckpoints && opts.debugLayers?.includes(l);

      if (isCheckpoint && currentRecorder) {
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
        currentRecorder = undefined;
      }

      const shouldDebug = opts.debug && currentHiddenBuffer && (!recorder || isCheckpoint);
      if (shouldDebug && !currentRecorder) {
        const device = getDevice();
        if (device) {
          if (allowReadback(`pipeline.prefill.layer-${l}`)) {
            try {
              const sampleSize = config.hiddenSize * activationBytes;
              const staging = device.createBuffer({
                size: sampleSize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
              });
              const enc = device.createCommandEncoder();
              const lastTokenOffset = (numTokens - 1) * config.hiddenSize * activationBytes;
              enc.copyBufferToBuffer(currentHiddenBuffer, lastTokenOffset, staging, 0, sampleSize);
              device.queue.submit([enc.finish()]);
              await staging.mapAsync(GPUMapMode.READ);
              const data = decodeReadback(staging.getMappedRange().slice(0), activationDtype);
              staging.unmap();
              staging.destroy();
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
    }

    if (benchmarkSubmits) {
      logSubmitStats(`Prefill (${numTokens} tokens, ${config.numLayers} layers)`);
      setTrackSubmits(false);
    }

    if (opts.debug) {
      log.debug('Pipeline', `LAYER_LOOP_DONE, currentHiddenBuffer type=${currentHiddenBuffer?.constructor?.name}`);
      if (currentHiddenBuffer && allowReadback('pipeline.prefill.final-hidden')) {
        const device = getDevice();
        const lastTokenOffset = (numTokens - 1) * config.hiddenSize * activationBytes;
        const sampleSize = config.hiddenSize * activationBytes;
        const staging = device.createBuffer({
          size: sampleSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(currentHiddenBuffer, lastTokenOffset, staging, 0, sampleSize);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = decodeReadback(staging.getMappedRange().slice(0), activationDtype);
        staging.unmap();
        staging.destroy();
        const nanCount = Array.from(data).filter(x => !Number.isFinite(x)).length;
        const nonZero = Array.from(data).filter(x => Number.isFinite(x) && x !== 0).slice(0, 5);
        log.debug('Pipeline', `FINAL_HIDDEN[pos=${numTokens - 1}]: nan=${nanCount}/${data.length}, sample=[${nonZero.map(x => x.toFixed(4)).join(', ')}]`);
      }
    }

    if (hasGpuTimePrefill) {
      this.#state.stats.gpuTimePrefillMs = gpuTimePrefillMs;
    }

    if (returnHidden) {
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

    
    let logits;
    let logitsVocabSize = config.vocabSize;
    let usedRecordedLogits = false;
    const lmHead = this.#state.weights.get('lm_head');
    const canRecordLogits = !!currentRecorder && !!lmHead && !isCpuWeightBuffer(lmHead) && !this.#state.disableRecordedLogits;
    if (currentRecorder && canRecordLogits) {
      const recorded = await recordLogitsGPU(
        currentRecorder,
        currentHiddenBuffer,
        numTokens,
        getLogitsWeights(this.#state),
        getLogitsConfig(this.#state)
      );
      logitsVocabSize = recorded.vocabSize;
      usedRecordedLogits = true;

      await currentRecorder.submitAndWait();
      await recordProfile(currentRecorder);

      const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: recorded.logitsDtype });
      const logitsData = await readBuffer(recorded.logitsBuffer, numTokens * logitsVocabSize * logitsBytes);
      releaseBuffer(recorded.logitsBuffer);
      logits = decodeReadback(logitsData, recorded.logitsDtype);

	      const health = getLogitsHealth(logits);
	      if (health.nanCount > 0 || health.infCount > 0 || health.nonZeroCount === 0) {
	        log.warn(
	          'Logits',
	          `Recorded logits invalid (nan=${health.nanCount} inf=${health.infCount} nonZero=${health.nonZeroCount}, maxAbs=${health.maxAbs.toFixed(3)}); recomputing without recorder.`
	        );
	        this.#state.disableRecordedLogits = true;
	        this.#state.disableFusedDecode = true;
	        logits = await computeLogits(
	          currentHiddenBuffer,
	          numTokens,
	          getLogitsWeights(this.#state),
	          getLogitsConfig(this.#state),
	          this.#state.useGPU,
	          this.#state.debugFlags,
	          undefined,
	          debugCheckBuffer,
	          this.#state.runtimeConfig.shared.debug.probes
	        );
	        const fallbackHealth = getLogitsHealth(logits);
	        if (fallbackHealth.nanCount > 0 || fallbackHealth.infCount > 0 || fallbackHealth.nonZeroCount === 0) {
	          throw new Error(
	            `[Logits] Fallback logits invalid (nan=${fallbackHealth.nanCount} inf=${fallbackHealth.infCount} nonZero=${fallbackHealth.nonZeroCount}, maxAbs=${fallbackHealth.maxAbs.toFixed(3)}). ` +
	            'This indicates upstream kernel output is NaN/Inf (often prefill attention/matmul).'
	          );
	        }
	        logitsVocabSize = config.vocabSize;
	        usedRecordedLogits = false;
	      }

      releaseBuffer(currentHiddenBuffer);
    } else {
      if (currentRecorder) {
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
      }
      logits = await computeLogits(
        currentHiddenBuffer,
        numTokens,
        getLogitsWeights(this.#state),
        getLogitsConfig(this.#state),
        this.#state.useGPU,
        this.#state.debugFlags,
        undefined,
        debugCheckBuffer,
        this.#state.runtimeConfig.shared.debug.probes
      );

      releaseBuffer(currentHiddenBuffer);
    }

    this.#state.currentSeqLen = startPos + numTokens;

    let lastLogits = extractLastPositionLogits(logits, numTokens, logitsVocabSize);
    if (usedRecordedLogits) {
      if (logitsVocabSize < config.vocabSize) {
        const padded = new Float32Array(config.vocabSize);
        padded.set(lastLogits);
        padded.fill(-Infinity, logitsVocabSize);
        lastLogits = padded;
      }
      if (config.finalLogitSoftcapping != null) {
        applySoftcapping(lastLogits, config.finalLogitSoftcapping);
      }
    }

    if (opts.debug) {
      logitsSanity(lastLogits, 'Prefill', (tokens) => this.#state.tokenizer?.decode?.(tokens) || '?');
    }

    if (opts.debug) {
      if (this.#state.kvCache?.hasGPUCache?.()) {
        log.debug('Pipeline', `KV cache active after prefill: seqLen=${this.#state.kvCache.getKeyCache(0)?.constructor.name ?? '?'}`);
      } else {
        log.warn('Pipeline', `KV cache NOT active after prefill! hasGPUCache=${this.#state.kvCache?.hasGPUCache?.()}`);
      }
    }

    return lastLogits;
  }

  
  async _decodeStep(currentIds, opts) {
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;
    return decodeStep(this.#state, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

  async decodeStepLogits(currentIds, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;

    return decodeStepLogits(this.#state, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

  async advanceWithToken(tokenId, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;

    this._assertTokenIdInRange(tokenId, 'advanceWithToken');
    await advanceWithToken(this.#state, tokenId, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

  async advanceWithTokenAndEmbedding(tokenId, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const modelType = String(this.#state.manifest?.modelType || '').toLowerCase();
    const configuredMode = this.#state.runtimeConfig.inference.generation.embeddingMode;
    const embeddingMode = options.embeddingMode ?? (modelType === 'embedding' ? 'mean' : configuredMode);
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;

    this._assertTokenIdInRange(tokenId, 'advanceWithTokenAndEmbedding');
    if (!advanceWithTokenAndEmbedding) {
      throw new Error(
        'advanceWithTokenAndEmbedding not available (likely stale module cache). ' +
        'Hard-reload the page to refresh @doppler/core.'
      );
    }

    return advanceWithTokenAndEmbedding(
      this.#state,
      tokenId,
      opts,
      this._getDecodeHelpers(debugCheckBuffer),
      embeddingMode
    );
  }

  async _generateNTokensGPU(startToken, N, currentIds, opts) {
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;
    return generateNTokensGPU(this.#state, startToken, N, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }
}
