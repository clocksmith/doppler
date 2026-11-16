import { prepareActivityRecording } from "../activation/activity-weights.js";
import { beginGenerationContext, beginSequenceMemoryBoundary, finishGenerationContext, generationContextInput } from "./sequence-state.js";
import { markWarmed as markKernelCacheWarmed } from "../../../../../gpu/kernel-selection-cache.js";
import { log, trace } from "../../../../../debug/index.js";
import { validateCallTimeOptions } from "../../../../../config/param-validator.js";
import { getTopK } from "../sampling/sampling.js";
import { isStopToken } from "../../init.js";
import { resolveGenerateOptions } from "./generator-runtime.js";
import { resolveTokenText } from "../../policy/generator-decode-policy.js";
import { getExperimentalIntentBundleModule } from "./text.js";
import { verifyDecodeStepAhead } from "../decode/decode.js";
import { wholeTokenProgramRequested } from "../whole-token/whole-token-lane.js";

// With a kept whole-token program, verify it on decode step 1 while the prefill's last
// commands run, so step 1 replays it. A failure there invalidates the program; step 1 then
// observes as it would without this.
function wholeTokenAheadVerification(generator, opts) {
  const state = generator._state;
  if (!state.wholeTokenProgram?.program.ready || !wholeTokenProgramRequested(state, opts)) return undefined;
  let used = false;
  return async (position, placeholderToken) => {
    if (used) return;
    used = true;
    try {
      await verifyDecodeStepAhead(state, position, placeholderToken, opts, generator._getDecodeHelpers(undefined));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      state.wholeTokenProgram?.program.invalidate(`verification ahead of decode step 1 failed: ${message}`);
      log.warn('Decode', `Whole-token verification ahead of step 1 failed (${message}); step 1 observes the program again.`);
    }
  };
}

async function samplePrefillToken(generator, prefillLogits, generatedIds, inputIds, opts, prefillStartSeqLen, warn) {
  try {
    return generator._sampleNextTokenFromLogits(prefillLogits, generatedIds, opts);
  } catch (error) {
    if (!generator._shouldUseFinitenessFallback(error, 'prefill-sample')) {
      throw error;
    }
    await finishGenerationContext(generator._state, [], false);
    generationContextInput(generator._state, inputIds);
    if (warn) log.warn('Pipeline', 'FinitenessGuard caught non-finite prefill logits at sampling. Retrying with F32 precision.');
    prefillLogits = await generator._retryWithPersistentFinitenessFallback(
      opts,
      'prefill-sample',
      opts.maxTokens,
      () => generator._prefill(inputIds, opts),
      opts.generationContext ? 0 : prefillStartSeqLen
    );
    return generator._sampleNextTokenFromLogits(prefillLogits, generatedIds, opts);
  }
}

export async function* _generateTokensInternal(prompt, options = {}, mode = 'text') {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    validateCallTimeOptions(options);
    const opts = resolveGenerateOptions(this._state, options);
    const startTime = this.beginGeneration(opts, options);
    let generatedIds = [];
    let promptTokenCount = 0;
    let completed = false;
    if (opts.debug) {
      log.debug('Pipeline', `ChatTemplate: options=${options.useChatTemplate}, final=${opts.useChatTemplate}`);
    }
    const emitToken = async function* (generator, tokenId, textDecoder) {
      if (mode === 'token') {
        yield tokenId;
        if (options.onToken) options.onToken(tokenId, '');
        return;
      }
      const tokenText = textDecoder(tokenId);
      yield tokenText;
      if (options.onToken) options.onToken(tokenId, tokenText);
    };
    try {
      beginGenerationContext(this._state, options.generationContext, opts);
      await prepareActivityRecording(this._state, opts);
      if (this._state.currentSeqLen === 0) await beginSequenceMemoryBoundary();
      const prefillStartSeqLen = this._state.currentSeqLen;
      const prefillStart = performance.now();
      opts._whileFinalPrefillRuns = wholeTokenAheadVerification(this, opts);
      const { inputIds, logits: initialPrefillLogits } = await this._prefillPromptToLogits(prompt, opts, 'generate');
      opts._whileFinalPrefillRuns = undefined;
      const prefillLogits = initialPrefillLogits;
      this._state.stats.prefillTimeMs = performance.now() - prefillStart;
      this._assertTokenIdsInRange(inputIds, 'generate.prefillTokens');
      generatedIds = [...inputIds];
      promptTokenCount = inputIds.length;
      this._state.stats.prefillTokens = opts.promptTokenCount = inputIds.length;
      if (opts.debug) {
        log.debug('Pipeline', `Input: ${inputIds.length} tokens`);
      }
      const intentBundleConfig = this._state.runtimeConfig.shared.intentBundle;
      const intentBundle = intentBundleConfig?.bundle;
      const expectedTopK = intentBundle?.payload?.expectedTopK
        ?? intentBundle?.payload?.expected_top_k;
      const maxDriftThreshold = intentBundle?.constraints?.maxDriftThreshold
        ?? intentBundle?.constraints?.max_drift_threshold;
      if (intentBundleConfig?.enabled && Array.isArray(expectedTopK) && expectedTopK.length > 0) {
        const { enforceLogitDrift } = await getExperimentalIntentBundleModule();
        const actualTopK = getTopK(
          prefillLogits,
          expectedTopK.length,
          (tokens) => resolveTokenText(this._state.tokenizer, tokens),
        ).map((token) => token.token);
        const driftResult = enforceLogitDrift(expectedTopK, actualTopK, maxDriftThreshold);
        if (!driftResult.ok) {
          throw new Error(`Intent bundle drift check failed: ${driftResult.reason}`);
        }
      }
      if (opts.debug) {
        const topAfterPenalty = getTopK(
          Float32Array.from(prefillLogits),
          5,
          (tokens) => resolveTokenText(this._state.tokenizer, tokens)
        );
        log.debug('Pipeline', `After rep penalty top-5: ${topAfterPenalty.map(t => `"${t.text}"(${(t.prob * 100).toFixed(1)}%)`).join(', ')}`);
      }

      const firstToken = await samplePrefillToken(this, prefillLogits, generatedIds, inputIds, opts, prefillStartSeqLen, true);

      if (opts.debug) {
        const firstTokenText = resolveTokenText(this._state.tokenizer, [firstToken], `[${firstToken}]`, (tokens) => this._state.tokenizer?.decode?.(tokens, true, false));
        log.debug('Pipeline', `First token sampled: id=${firstToken} text="${firstTokenText}"`);
      }

      const stopTokenIds = this._state.modelConfig.stopTokenIds;
      const eosToken = this._state.tokenizer.getSpecialTokens?.()?.eos;
      const stopSequenceStart = inputIds.length;
      generatedIds.push(firstToken);
      this._state.stats.ttftMs = performance.now() - startTime;

      const decodeToken = (tokenId) => resolveTokenText(
        this._state.tokenizer,
        [tokenId],
        `[${tokenId}]`,
        (tokens) => this._state.tokenizer?.decode?.(tokens, true, false),
        (tokens) => this._state.tokenizer?.decode?.(tokens, false, false)
      );
      const decodeRuntime = {
        stopTokenIds,
        eosToken,
        stopSequenceStart,
        decodeToken,
        logBatchPath: opts.debug,
        emitMode: mode,
      };

      yield* emitToken(this, firstToken, decodeToken);

      if (options.signal?.aborted || this._shouldStopAfterAppendedToken(generatedIds, firstToken, opts, decodeRuntime)) {
        this._recordStopReason(options.signal?.aborted ? 'aborted'
          : isStopToken(firstToken, decodeRuntime.stopTokenIds, decodeRuntime.eosToken) ? 'stop-token' : 'stop-sequence',
          options.signal?.aborted ? null : firstToken);
        this._state.stats.decodeTimeMs = 0;
        this._state.stats.tokensGenerated = 1;
        this._state.stats.decodeTokens = 0;
      } else {
        yield* this._runDecodeLoop(generatedIds, opts, options, decodeRuntime);
      }
      const tokensGenerated = this._state.stats.tokensGenerated ?? 1;
      this._state.stats.totalTimeMs = performance.now() - startTime;

      if (opts.debug) {
        log.debug('Pipeline', `Generated ${tokensGenerated} tokens in ${this._state.stats.totalTimeMs.toFixed(0)}ms`);
      }

      const ttft = this._state.stats.ttftMs ?? this._state.stats.prefillTimeMs;
      const decodeTokens = Math.max(0, tokensGenerated - 1);
      const decodeSpeed = decodeTokens > 0 ? (decodeTokens / this._state.stats.decodeTimeMs * 1000) : 0;
      const loadMs = this._state.stats.modelLoadMs;
      const loadLabel = Number.isFinite(loadMs) ? `Load: ${loadMs.toFixed(0)}ms | ` : '';
      if (opts.benchmark) {
        log.info('Benchmark', `${loadLabel}TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this._state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this._state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      } else {
        log.info('Perf', `${loadLabel}TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this._state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this._state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      }
      trace.perf('Decode summary', {
        ttftMs: ttft,
        prefillMs: this._state.stats.prefillTimeMs,
        decodeMs: this._state.stats.decodeTimeMs,
        decodeTokens,
        decodeSpeed,
        totalMs: this._state.stats.totalTimeMs,
      });
      completed = !options.signal?.aborted && !this._finitenessFallbackWindow;
    } finally {
      try { await opts.__activationTrace?.drain(); }
      finally {
        try { this._closeFinitenessFallbackWindow(opts); }
        finally {
          try { await finishGenerationContext(this._state, generatedIds, completed); }
          finally { this.finishGeneration(options, startTime, generatedIds, promptTokenCount); }
        }
      }
    }
  }

export async function generateTokenIds(prompt, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);
    const opts = resolveGenerateOptions(this._state, options);
    const startTime = this.beginGeneration(opts, options);
    let generatedIds = [];
    let promptTokenCount = 0;
    let completed = false;

    try {
      beginGenerationContext(this._state, options.generationContext, opts);
      await prepareActivityRecording(this._state, opts);
      if (this._state.currentSeqLen === 0) await beginSequenceMemoryBoundary();
      const prefillStartSeqLen = this._state.currentSeqLen;
      const prefillStart = performance.now();
      const { inputIds, logits: initialPrefillLogits } = await this._prefillPromptToLogits(prompt, opts, 'generateTokenIds');
      const prefillLogits = initialPrefillLogits;
      this._state.stats.prefillTimeMs = performance.now() - prefillStart;
      this._assertTokenIdsInRange(inputIds, 'generateTokenIds.prefillTokens');
      generatedIds = [...inputIds];
      promptTokenCount = inputIds.length;
      this._state.stats.prefillTokens = opts.promptTokenCount = inputIds.length;

      const firstToken = await samplePrefillToken(this, prefillLogits, generatedIds, inputIds, opts, prefillStartSeqLen, false);

      const stopTokenIds = this._state.modelConfig.stopTokenIds;
      const eosToken = this._state.tokenizer.getSpecialTokens?.()?.eos;
      const stopSequenceStart = inputIds.length;
      generatedIds.push(firstToken);
      const tokenIds = [firstToken];
      this._state.stats.ttftMs = performance.now() - startTime;
      markKernelCacheWarmed();
      if (options.onToken) options.onToken(firstToken, '');

      const decodeRuntime = {
        stopTokenIds,
        eosToken,
        stopSequenceStart,
        decodeToken: () => '',
        emitMode: 'token',
      };

      if (!options.signal?.aborted && !this._shouldStopAfterAppendedToken(generatedIds, firstToken, opts, decodeRuntime)) {
        for await (const tokenId of this._runDecodeLoop(generatedIds, opts, options, decodeRuntime)) {
          tokenIds.push(tokenId);
        }
      } else {
        this._recordStopReason(options.signal?.aborted ? 'aborted'
          : isStopToken(firstToken, decodeRuntime.stopTokenIds, decodeRuntime.eosToken) ? 'stop-token' : 'stop-sequence',
          options.signal?.aborted ? null : firstToken);
        this._state.stats.decodeTimeMs = 0;
        this._state.stats.tokensGenerated = 1;
        this._state.stats.decodeTokens = 0;
      }

      this._state.stats.totalTimeMs = performance.now() - startTime;

      completed = !options.signal?.aborted && !this._finitenessFallbackWindow;
      return {
        tokenIds,
        // Exact prefilled IDs, so callers can replay a prefix without
        // re-tokenizing displayed text.
        inputTokenIds: [...inputIds],
        stats: this._state.stats,
      };
    } finally {
      try { await opts.__activationTrace?.drain(); }
      finally {
        try { this._closeFinitenessFallbackWindow(opts); }
        finally {
          try { await finishGenerationContext(this._state, generatedIds, completed); }
          finally { this.finishGeneration(options, startTime, generatedIds, promptTokenCount); }
        }
      }
    }
  }
