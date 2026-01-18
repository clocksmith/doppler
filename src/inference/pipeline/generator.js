

import { getDevice, setTrackSubmits } from '../../gpu/device.js';
import { releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { isGPUSamplingAvailable } from '../../gpu/kernels/sample.js';
import { markWarmed as markKernelCacheWarmed } from '../../gpu/kernel-selection-cache.js';
import { resetSubmitStats, logSubmitStats } from '../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../gpu/command-recorder.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import { log } from '../../debug/index.js';
import { validateCallTimeOptions } from '../../config/param-validator.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

// Pipeline sub-modules
import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from './sampling.js';
import { enforceLogitDrift } from '../../hotswap/intent-bundle.js';
import { applyChatTemplate, isStopToken } from './init.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { computeLogits, recordLogitsGPU, extractLastPositionLogits, applySoftcapping } from './logits.js';
import { isWeightBuffer, isCpuWeightBuffer, getWeightDtype } from '../../gpu/weight-buffer.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { decodeStep, generateNTokensGPU, sumProfileTimings } from './generator-steps.js';
import { buildLayerContext, debugCheckBuffer as debugCheckBufferHelper, getLogitsConfig, getLogitsWeights } from './generator-helpers.js';

import { decodeReadback, getLogitsHealth } from './debug-utils.js';

export class PipelineGenerator {
  
  #state;

  
  constructor(state) {
    this.#state = state;
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
    this.#state.stats.gpuTimePrefillMs = undefined;
    this.#state.stats.gpuTimeDecodeMs = undefined;
    const startTime = performance.now();

    const runtimeDefaults = this.#state.runtimeConfig.inference;
    const samplingDefaults = runtimeDefaults.sampling;
    const batchingDefaults = runtimeDefaults.batching;

    const opts = {
      maxTokens: options.maxTokens ?? batchingDefaults.maxTokens,
      temperature: options.temperature ?? samplingDefaults.temperature,
      topP: options.topP ?? samplingDefaults.topP,
      topK: options.topK ?? samplingDefaults.topK,
      repetitionPenalty: options.repetitionPenalty ?? samplingDefaults.repetitionPenalty,
      stopSequences: options.stopSequences ?? [],
      useSpeculative: options.useSpeculative ?? false,
      useChatTemplate: options.useChatTemplate
        ?? this.#state.runtimeConfig.inference.chatTemplate?.enabled
        ?? this.#state.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? false,
      benchmark: options.benchmark ?? false,
      disableCommandBatching: options.disableCommandBatching ?? false,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? false,
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
      const generatedIds = [...inputIds];

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

      const firstText = this.#state.tokenizer.decode([firstToken], true, false);
      yield firstText;
      if (options.onToken) options.onToken(firstToken, firstText);

      const stopTokenIds = this.#state.modelConfig.stopTokenIds || [];
      const eosToken = this.#state.tokenizer.getSpecialTokens?.()?.eos;
      let tokensGenerated = 1;

      markKernelCacheWarmed();

      const decodeStart = performance.now();
      const useBatchPath = opts.batchSize > 1
        && this.#state.useGPU
        && isGPUSamplingAvailable()
        && !opts.disableMultiTokenDecode
        && !opts.disableCommandBatching;

      if (opts.debug && useBatchPath) {
        log.debug('Pipeline', `Using batch decode path with batchSize=${opts.batchSize}, stopCheckMode=${opts.stopCheckMode}`);
      }

      while (tokensGenerated < opts.maxTokens) {
        if (options.signal?.aborted) break;

        if (useBatchPath) {
          const remaining = opts.maxTokens - tokensGenerated;
          const thisBatchSize = Math.min(opts.batchSize, remaining);
          const lastToken = generatedIds[generatedIds.length - 1];

          try {
            const batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, opts);

            
            const batchTokens = [];
            for (const tokenId of batchResult.tokens) {
              generatedIds.push(tokenId);
              tokensGenerated++;

              const tokenText = this.#state.tokenizer.decode([tokenId], true, false);
              yield tokenText;
              if (options.onToken) options.onToken(tokenId, tokenText);
              batchTokens.push({ id: tokenId, text: tokenText });
            }

            if (options.onBatch) options.onBatch(batchTokens);

            if (batchResult.actualCount < thisBatchSize) {
              break;
            }

            if (opts.stopSequences.length > 0) {
              const fullText = this.#state.tokenizer.decode(generatedIds.slice(inputIds.length), false);
              if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
            }
          } catch (error) {
            log.warn('Pipeline', `Batch decode failed, falling back to single-token: ${error}`);
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
            const fullText = this.#state.tokenizer.decode(generatedIds.slice(inputIds.length), false);
            if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
          }
        }
      }

      this.#state.stats.decodeTimeMs = performance.now() - decodeStart;
      this.#state.stats.tokensGenerated = tokensGenerated;
      this.#state.stats.totalTimeMs = performance.now() - startTime;

      if (opts.debug) {
        log.debug('Pipeline', `Generated ${tokensGenerated} tokens in ${this.#state.stats.totalTimeMs.toFixed(0)}ms`);
      }

      if (opts.benchmark) {
        const ttft = this.#state.stats.prefillTimeMs;
        const decodeTokens = tokensGenerated - 1;
        const decodeSpeed = decodeTokens > 0 ? (decodeTokens / this.#state.stats.decodeTimeMs * 1000) : 0;
        log.info('Benchmark', `TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this.#state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this.#state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      }
    } finally {
      this.#state.isGenerating = false;
    }
  }

  
  async prefillKVOnly(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    this.#state.stats.gpuTimePrefillMs = undefined;

    const opts = {
      useChatTemplate: options.useChatTemplate
        ?? this.#state.runtimeConfig.inference.chatTemplate?.enabled
        ?? this.#state.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? false,
      disableCommandBatching: options.disableCommandBatching ?? false,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? false,
    };

    let processedPrompt = prompt;
    if (opts.useChatTemplate && this.#state.modelConfig.chatTemplateType) {
      processedPrompt = applyChatTemplate(prompt, this.#state.modelConfig.chatTemplateType);
    }

    const inputIds = this.#state.tokenizer.encode(processedPrompt);
    if (opts.debug) {
      log.debug('Pipeline', `PrefillKVOnly: ${inputIds.length} tokens`);
    }

    await this._prefill(inputIds, opts);

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
    const startTime = performance.now();

    const runtimeDefaults = this.#state.runtimeConfig.inference;
    const samplingDefaults = runtimeDefaults.sampling;
    const batchingDefaults = runtimeDefaults.batching;

    const opts = {
      maxTokens: options.maxTokens ?? batchingDefaults.maxTokens,
      temperature: options.temperature ?? samplingDefaults.temperature,
      topP: options.topP ?? samplingDefaults.topP,
      topK: options.topK ?? samplingDefaults.topK,
      repetitionPenalty: options.repetitionPenalty ?? samplingDefaults.repetitionPenalty,
      stopSequences: options.stopSequences ?? [],
      useSpeculative: options.useSpeculative ?? false,
      useChatTemplate: options.useChatTemplate
        ?? this.#state.runtimeConfig.inference.chatTemplate?.enabled
        ?? this.#state.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? this.#state.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? false,
      benchmark: options.benchmark ?? false,
      disableCommandBatching: options.disableCommandBatching ?? false,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? false,
      batchSize: options.batchSize ?? batchingDefaults.batchSize,
      stopCheckMode: options.stopCheckMode ?? batchingDefaults.stopCheckMode,
    };

    try {
      let processedPrompt = prompt;
      if (opts.useChatTemplate && this.#state.modelConfig.chatTemplateType) {
        processedPrompt = applyChatTemplate(prompt, this.#state.modelConfig.chatTemplateType);
      }

      const inputIds = this.#state.tokenizer.encode(processedPrompt);
      const generatedIds = [...prefix.tokens, ...inputIds];
      const promptTokenCount = generatedIds.length;

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

      const firstText = this.#state.tokenizer.decode([firstToken], true, false);
      yield firstText;
      if (options.onToken) options.onToken(firstToken, firstText);

      const stopTokenIds = this.#state.modelConfig.stopTokenIds || [];
      const eosToken = this.#state.tokenizer.getSpecialTokens?.()?.eos;
      let tokensGenerated = 1;

      markKernelCacheWarmed();

      const decodeStart = performance.now();
      const useBatchPath = opts.batchSize > 1
        && this.#state.useGPU
        && isGPUSamplingAvailable()
        && !opts.disableMultiTokenDecode
        && !opts.disableCommandBatching;

      while (tokensGenerated < opts.maxTokens) {
        if (options.signal?.aborted) break;

        if (useBatchPath) {
          const remaining = opts.maxTokens - tokensGenerated;
          const thisBatchSize = Math.min(opts.batchSize, remaining);
          const lastToken = generatedIds[generatedIds.length - 1];

          try {
            const batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, opts);
            
            const batchTokens = [];
            for (const tokenId of batchResult.tokens) {
              generatedIds.push(tokenId);
              tokensGenerated++;
              const tokenText = this.#state.tokenizer.decode([tokenId], true, false);
              yield tokenText;
              if (options.onToken) options.onToken(tokenId, tokenText);
              batchTokens.push({ id: tokenId, text: tokenText });
            }
            if (options.onBatch) options.onBatch(batchTokens);
            if (batchResult.actualCount < thisBatchSize) break;
            if (opts.stopSequences.length > 0) {
              const fullText = this.#state.tokenizer.decode(generatedIds.slice(promptTokenCount), false);
              if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
            }
          } catch (error) {
            log.warn('Pipeline', `Batch decode failed, falling back to single-token: ${error}`);
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
      this.#state.stats.totalTimeMs = performance.now() - startTime;
    } finally {
      this.#state.isGenerating = false;
    }
  }

  // ==========================================================================
  // Internal Methods (Prefill, Decode, Helpers)
  // ==========================================================================

  
  async _prefill(inputIds, opts) {
    const numTokens = inputIds.length;
    const config = this.#state.modelConfig;
    const startPos = this.#state.currentSeqLen;
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
    return decodeStep(this.#state, currentIds, opts, {
      buildLayerContext: (recorder, isDecodeMode, debugLayers) =>
        buildLayerContext(this.#state, recorder, isDecodeMode, debugLayers, debugCheckBuffer),
      getLogitsWeights: () => getLogitsWeights(this.#state),
      getLogitsConfig: () => getLogitsConfig(this.#state),
      debugCheckBuffer,
    });
  }

  async _generateNTokensGPU(startToken, N, currentIds, opts) {
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;
    return generateNTokensGPU(this.#state, startToken, N, currentIds, opts, {
      buildLayerContext: (recorder, isDecodeMode, debugLayers) =>
        buildLayerContext(this.#state, recorder, isDecodeMode, debugLayers, debugCheckBuffer),
      getLogitsWeights: () => getLogitsWeights(this.#state),
      getLogitsConfig: () => getLogitsConfig(this.#state),
    });
  }
}
