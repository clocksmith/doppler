import { beginPrefillDiagnostics, closePrefillDiagnostics } from "../session/diagnostic-sequence.js";
import { resolvePrefillOptions } from "../session/generator-runtime.js";
import { assertIncrementalDecodeSupport } from "../../policy/generator-decode-policy.js";
import { normalizeSelectedLogitTokenIds } from "../session/text.js";
import { selectedTokenLogSoftmax } from "../../logits/selected-log-softmax.js";

// Teacher-forced target scoring: one prefill over prompt + target[0..T-2]; the LM head
// projects only the T trailing rows, each predicting the next target token. Log-softmax
// runs over each complete vocabulary row after readback; nothing is approximated.
/** @type {import('./target-scoring.js').prefillWithTargetTokenLogProbs} */
export async function prefillWithTargetTokenLogProbs(prompt, targetTokenIds, options) {
  if (!this._state.isLoaded) throw new Error('Model not loaded');
  if (this._state.isGenerating && options.__internalGenerate !== true) {
    throw new Error('Generation already in progress');
  }
  assertIncrementalDecodeSupport(this._state, 'prefillWithTargetTokenLogProbs');
  const totalStart = performance.now();
  const vocabSize = this._state.modelConfig.vocabSize;
  const targets = normalizeSelectedLogitTokenIds(
    targetTokenIds,
    vocabSize,
    'prefillWithTargetTokenLogProbs.targetTokenIds'
  );
  if (targets === null) throw new Error('prefillWithTargetTokenLogProbs.targetTokenIds is required.');
  if (!Array.isArray(options.inputIds) || options.inputIds.length < 1) {
    throw new Error('prefillWithTargetTokenLogProbs requires explicit prompt inputIds.');
  }
  const promptIds = Array.from(options.inputIds, Number);
  this._assertTokenIdsInRange(promptIds, 'prefillWithTargetTokenLogProbs.inputIds');
  const sequence = [...promptIds, ...targets.slice(0, -1)];
  // log P(target | prompt) is a fresh-context quantity. _prefill starts at currentSeqLen, so a
  // finished generation's KV, recurrent and conv state would otherwise condition the score.
  this._resetReplayPrefillRuntimeState();
  this._resetDecodeRuntimeState();
  this._state.stats.gpuTimePrefillMs = undefined;
  this._state.stats.prefillProfileSteps = [];
  const opts = { ...resolvePrefillOptions(this._state, { ...options, inputIds: sequence }), _trailingLogitRows: targets.length };
  try {
    beginPrefillDiagnostics(this._state, options, opts.executionPlan);
    const { inputIds, logits, phase: prefillPhase } = await this._prefillPromptToLogits(
      prompt, opts, 'prefillWithTargetTokenLogProbs'
    );
    if (!(logits instanceof Float32Array) || logits.length !== targets.length * vocabSize) {
      throw new Error(
        `prefillWithTargetTokenLogProbs expected ${targets.length} full logit rows of ${vocabSize}; ` +
        `got ${logits instanceof Float32Array ? logits.length : 'no'} values.`
      );
    }
    const scored = selectedTokenLogSoftmax(logits, vocabSize, targets);
    return {
      seqLen: this._state.currentSeqLen,
      tokens: inputIds.slice(0, promptIds.length),
      targetTokenIds: targets,
      logits: scored.logits,
      logSumExp: scored.logSumExp,
      logProbs: scored.logProbs,
      phase: {
        ...(prefillPhase ?? {}),
        totalMs: performance.now() - totalStart,
        targetTokenCount: targets.length,
      },
    };
  } finally {
    closePrefillDiagnostics(this, opts);
  }
}
