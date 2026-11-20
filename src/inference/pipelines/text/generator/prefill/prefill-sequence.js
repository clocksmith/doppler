import { getDevice } from "../../../../../gpu/device.js";
import { generationContextInput, prepareGenerationContextPrefill, finishGenerationContext } from "../session/sequence-state.js";
import { releaseBuffer } from "../../../../../memory/buffer-pool.js";
import { log } from "../../../../../debug/index.js";
import { readPrefillExpertCounters } from "../../expert-loading-stats.js";
import { resolveEffectivePrefillTokenChunkSize } from "./prefill-policy.js";
import { snapshotPrefillPhase } from "./prefill-observation.js";
import { recordPrefillRecorderStats } from "../session/text.js";

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
    const inputStart = performance.now();
    const inputIds = this._resolvePromptOrInputIds(prompt, opts.useChatTemplate, contextLabel, opts.inputIds);
    generationContextInput(this._state, inputIds);
    const prefillStartSeqLen = this._state.currentSeqLen;
    const inputMs = performance.now() - inputStart;
    if (opts.debug) {
      log.debug('Pipeline', `${contextLabel}: ${inputIds.length} tokens`);
    }
    const loader = this._state.dopplerLoader;
    const routed = this._state.runtimeConfig?.shared?.debug?.profiler?.enabled === true
      ? loader?.getRoutedExpertPolicy?.() : null;
    const before = routed?.enabled ? readPrefillExpertCounters(loader) : null;
    let logits;
    const prefillStart = performance.now();
    const runPrefill = async () => {
      const computedIds = generationContextInput(this._state, inputIds);
      const checkpoint = await prepareGenerationContextPrefill(this, prompt, inputIds, opts);
      return this._prefillInputIdsToLogits(checkpoint === null ? computedIds : inputIds.slice(checkpoint), opts);
    };
    try {
      logits = await runPrefill();
    } catch (error) {
      if (!this._shouldUseFinitenessFallback(error, contextLabel)) {
        throw error;
      }
      await finishGenerationContext(this._state, [], false);
      log.warn('Pipeline', `FinitenessGuard caught NaN/Inf during ${contextLabel}. Retrying with F32 precision.`);
      logits = await this._retryWithPersistentFinitenessFallback(
        opts,
        contextLabel,
        opts.maxTokens ?? 1,
        runPrefill,
        opts.generationContext ? 0 : prefillStartSeqLen
      );
    }
    const prefillMs = performance.now() - prefillStart;
    if (before) {
      const after = readPrefillExpertCounters(loader);
      const deltas = Object.fromEntries(Object.entries(after).map(([key, value]) =>
        [key, value === null ? null : value - (before[key] ?? 0)]));
      log.info('Perf', 'Routed prefill', {
        inputMs, prefillMs, promptTokens: inputIds.length,
        computedTokens: this._state.stats.prefillComputedTokens,
        reusedTokens: this._state.stats.reusedPrefixTokens,
        tokenChunkLimit: resolveEffectivePrefillTokenChunkSize(this._state) ?? null,
        routeChunkLimit: routed.routeChunkTokens,
        ...deltas,
      });
    }
    return { inputIds, logits, phase: snapshotPrefillPhase(this._state.stats, {
      inputMs, prefillMs, tokens: inputIds.length,
    }) };
  }

export async function _prefillToHidden(inputIds, opts) {
    // Internal-only: reuse the main prefill implementation but stop before logits.
    return this._prefill(inputIds, { ...opts, _returnHidden: true });
  }
