import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { scoreTargetSequence, tokenizeTargetContinuation } from '../../src/inference/target-sequence-scoring.js';
import {
  TARGET_SEQUENCE_SCORING_CONTRACT, roundTargetSequenceLogProb, snapshotTargetSequenceScoringRequest,
  validateTargetSequenceScoreResult,
} from '../../src/config/target-sequence-scoring.js';
import { selectedTokenLogSoftmax } from '../../src/inference/pipelines/text/logits/selected-log-softmax.js';
import { prefillWithTargetTokenLogProbs } from '../../src/inference/pipelines/text/generator/adapters/target-scoring.js';
import { _prefillInputIdsToLogits } from '../../src/inference/pipelines/text/generator/prefill/prefill-runtime.js';
import { resolvePrefillTokenChunkSize } from '../../src/inference/pipelines/text/generator/prefill/prefill-policy.js';
import { formatChatMessages } from '../../src/inference/pipelines/text/generator/request/chat-format.js';

// --- Log-softmax math against a direct float64 computation. ---
const vocab = 7;
const rows = Float32Array.from([
  0.5, -1.25, 3, 2.75, -8, 0, 1,
  1000, 999.5, -1000, 998, 1000, -Infinity, 0, // exp() of raw values would overflow
  -3, -3, -3, -3, -3, -3, -3,
]);
const picks = [2, 1, 6];
const direct = picks.map((token, row) => {
  const values = Array.from(rows.subarray(row * vocab, (row + 1) * vocab));
  const shift = row === 1 ? 1000 : 0; // exact analytic shift keeps the reference finite
  const denominator = values.reduce((sum, value) => sum + Math.exp(value - shift), 0);
  return Math.log(Math.exp(values[token] - shift) / denominator);
});
const scored = selectedTokenLogSoftmax(rows, vocab, picks);
for (const [index, value] of scored.logProbs.entries()) {
  assert.ok(Math.abs(value - direct[index]) < 1e-12, `row ${index}: ${value} vs ${direct[index]}`);
  assert.ok(value <= 0);
  assert.equal(value, scored.logits[index] - scored.logSumExp[index]);
}
assert.ok(Math.abs(scored.logProbs[2] - Math.log(1 / 7)) < 1e-12, 'uniform row is log(1/V)');
assert.deepEqual(selectedTokenLogSoftmax(rows, vocab, picks), scored, 'reproducible bit-for-bit');
const other = selectedTokenLogSoftmax(rows, vocab, [5, 1, 6]);
assert.ok(Math.abs((other.logProbs[0] - scored.logProbs[0]) - (rows[5] - rows[2])) < 1e-12,
  'within a row, log-probabilities differ exactly by the logit difference');
assert.equal(selectedTokenLogSoftmax(Float32Array.from([0, -Infinity]), 2, [1]).logProbs[0], -Infinity);
assert.throws(() => selectedTokenLogSoftmax(Float32Array.from([0, NaN]), 2, [0]), /NaN/);
assert.throws(() => selectedTokenLogSoftmax(Float32Array.from([-Infinity, -Infinity]), 2, [0]), /no finite/);
assert.throws(() => selectedTokenLogSoftmax(rows, vocab, [0, 1]), /complete rows/);
assert.throws(() => selectedTokenLogSoftmax(rows, vocab, [7, 0, 0]), /outside/);

// --- Generator adapter: one prefill over prompt + target[0..T-2], T trailing rows. ---
function adapterContext({ reject = null } = {}) {
  const calls = [];
  const state = {
    isLoaded: true, isGenerating: false, useGPU: false, currentSeqLen: 0,
    modelConfig: { vocabSize: vocab, layerTypes: [], chatTemplateEnabled: false },
    manifest: { modelId: 'target-scoring-fixture' },
    runtimeConfig: createDopplerConfig().runtime,
    executionPlanState: { activePlanId: 'primary', primaryPlan: { id: 'primary', defaultDisableCommandBatching: false, defaultDisableMultiTokenDecode: false } },
    stats: {}, operatorDiagnostics: null,
  };
  return {
    calls,
    _state: state,
    _resetDecodeRuntimeState() {},
    _assertTokenIdsInRange(ids) { assert.ok(ids.every(id => id >= 0 && id < vocab)); },
    _closeFinitenessFallbackWindow() {},
    async _prefillPromptToLogits(prompt, opts) {
      calls.push({ prompt, inputIds: opts.inputIds, trailing: opts._trailingLogitRows });
      if (reject) throw reject;
      state.currentSeqLen = opts.inputIds.length;
      return { inputIds: opts.inputIds, logits: rows, phase: {} };
    },
  };
}
const adapter = adapterContext();
const adapted = await prefillWithTargetTokenLogProbs.call(adapter, 'p', picks, { inputIds: [4, 5] });
assert.deepEqual(adapter.calls, [{ prompt: 'p', inputIds: [4, 5, 2, 1], trailing: 3 }],
  'the last target token is never fed: its logits would predict past the target');
assert.deepEqual(adapted.tokens, [4, 5]);
assert.deepEqual(adapted.targetTokenIds, picks);
assert.deepEqual(adapted.logProbs, scored.logProbs);
await assert.rejects(prefillWithTargetTokenLogProbs.call(adapterContext(), 'p', picks, {}), /explicit prompt inputIds/);
await assert.rejects(prefillWithTargetTokenLogProbs.call(adapterContext(), 'p', [9], { inputIds: [1] }), /\[0, 7\)/);
await assert.rejects(prefillWithTargetTokenLogProbs.call(adapterContext(), 'p', [1, 2], { inputIds: [1] }), /expected 2 full logit rows/);
const deviceLost = new Error('device lost');
await assert.rejects(prefillWithTargetTokenLogProbs.call(adapterContext({ reject: deviceLost }), 'p', picks, { inputIds: [1] }),
  error => error === deviceLost);

// --- Token-chunked prefill keeps every trailing row in the final (logits) chunk. ---
async function chunkPlan(count, chunkSize, trailing) {
  const calls = [];
  const policy = { runtimeConfig: { inference: { session: { prefillTokenChunkSize: chunkSize } } }, modelConfig: {} };
  const context = {
    _resolvePrefillTokenChunkSize: ids => resolvePrefillTokenChunkSize(policy, ids.length),
    _prefillToHidden: async ids => { calls.push(ids.length); return ids.length; },
    _commitPrefillHiddenChunk: async () => {},
    _prefill: async ids => { calls.push(ids.length); return 'final'; },
  };
  await _prefillInputIdsToLogits.call(context, Array.from({ length: count }, (_, i) => i), { _trailingLogitRows: trailing });
  return calls;
}
assert.deepEqual(await chunkPlan(10, 4, undefined), [4, 4, 2], 'ordinary prefill chunking is unchanged');
assert.deepEqual(await chunkPlan(10, 4, 3), [4, 3, 3]);
assert.deepEqual(await chunkPlan(9, 4, 4), [4, 1, 4]);
assert.deepEqual(await chunkPlan(12, 4, 4), [4, 4, 4]);
await assert.rejects(chunkPlan(10, 4, 5), /trailing rows/);

// --- Scorer: tokenization boundary checks, totals, reset discipline, abort. ---
// Character tokenizer with BOS=1 and one merge ("\n\n" -> 2), enough to provoke a boundary merge.
const tokenizer = { encode(text) {
  const ids = [1];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n' && text[i + 1] === '\n') { ids.push(2); i++; } else ids.push(text.charCodeAt(i));
  }
  return ids;
} };
let prefills = 0, resets = 0, failure = null, during = null, lastCall = null;
const pipeline = {
  tokenizer,
  modelConfig: { chatTemplateType: 'qwen' },
  async prefillWithTargetTokenLogProbs(prompt, targetTokenIds, options) {
    prefills++;
    lastCall = { prompt, targetTokenIds, options };
    assert.equal(options.useChatTemplate, false);
    during?.();
    if (failure) throw failure;
    return { tokens: options.inputIds, targetTokenIds, logProbs: targetTokenIds.map((_, i) => -0.125 * (i + 1)) };
  },
  resetGenerationState() { resets++; },
};
const request = { prompt: 'Q:', target: ' hi', maxSeqLen: 16 };
const result = await scoreTargetSequence(pipeline, request);
assert.equal(result.schema, TARGET_SEQUENCE_SCORING_CONTRACT.resultSchema);
assert.deepEqual(result.targetTokenIds, [32, 104, 105]);
assert.deepEqual(lastCall.options.inputIds, [1, 81, 58]);
assert.equal(result.promptTokenCount, 3);
assert.equal(result.tokenCount, 3);
assert.deepEqual(result.tokenLogProbs, [-0.125, -0.25, -0.375]);
assert.equal(result.totalLogProb, -0.75);
assert.equal(result.meanLogProb, -0.25);
assert.equal(result.roundedTotalLogProb, -0.75);
assert.deepEqual(validateTargetSequenceScoreResult(request, result), result);
assert.equal(resets, 1);
for (const bad of [
  { ...result, totalLogProb: -0.7 }, { ...result, roundedTotalLogProb: -0.8 },
  { ...result, tokenLogProbs: [-0.125, -0.25, 0.375] }, { ...result, tokenLogProbs: [-0.125, -0.25, NaN] },
  { ...result, schema: 'doppler.choice-scores/v1' }, { ...result, tokenCount: 2 },
]) assert.throws(() => validateTargetSequenceScoreResult(request, bad));

// Rounding is a fixed contract, not locale or formatting dependent.
assert.equal(roundTargetSequenceLogProb(-12.344999), -12.34);
assert.equal(roundTargetSequenceLogProb(-12.345001), -12.35);
assert.equal(roundTargetSequenceLogProb(-0.004), 0);
assert.ok(Object.is(roundTargetSequenceLogProb(-0.004), 0), 'no negative zero');

// Messages use the model template in non-thinking mode, formatted exactly once.
const messages = [{ role: 'system', content: 'Reply tersely.' }, { role: 'user', content: 'Say hi' }];
await scoreTargetSequence(pipeline, { messages, target: 'hi', maxSeqLen: 256 });
const expectedPrompt = formatChatMessages(messages, 'qwen', { thinking: false });
assert.equal(lastCall.prompt, expectedPrompt);
assert.ok(expectedPrompt.endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n'), 'non-thinking assistant turn');
await assert.rejects(scoreTargetSequence({ ...pipeline, modelConfig: { chatTemplateType: null } },
  { messages, target: 'hi', maxSeqLen: 256 }), /chat template/);

const callsBefore = prefills;
for (const [bad, pattern] of [
  [{ prompt: 'Q:\n', target: '\nX', maxSeqLen: 16 }, /merges across the prompt boundary/],
  [{ prompt: 'Q:', target: 'x'.repeat(TARGET_SEQUENCE_SCORING_CONTRACT.maxTargetTokens + 1), maxSeqLen: 64 }, /at most/],
  [{ prompt: 'Q:', target: ' hi', maxSeqLen: 5 }, /exceeds maxSeqLen/],
  [{ prompt: 'Q:', target: '', maxSeqLen: 16 }, /non-empty target/],
  [{ prompt: 'Q:', messages, target: 'x', maxSeqLen: 16 }, /exactly one/],
  [{ target: 'x', maxSeqLen: 16 }, /exactly one/],
  [{ messages: [{ role: 'assistant', content: 'x' }], target: 'x', maxSeqLen: 16 }, /role/],
  [{ messages: [{ role: 'system', content: 'x' }], target: 'x', maxSeqLen: 16 }, /user message/],
  [{ ...request, useChatTemplate: true }, /accepts only/],
]) await assert.rejects(scoreTargetSequence(pipeline, bad), pattern);
assert.equal(prefills, callsBefore, 'invalid requests and boundaries fail before GPU execution');
assert.throws(() => tokenizeTargetContinuation({ tokenizer: { encode: text => text === 'a' ? [1, 2] : [1, 3, 4] } }, 'a', 'b', 8),
  /changes at token 1/);
assert.throws(() => tokenizeTargetContinuation({ tokenizer: { encode: () => [1, 2] } }, 'a', 'b', 8), /adds no tokens/);

const snapshot = snapshotTargetSequenceScoringRequest({ messages, target: ' t', maxSeqLen: 8 });
assert.ok(Object.isFrozen(snapshot.messages[0]));
assert.equal(snapshot.target, ' t', 'targets are exact, never trimmed');

failure = new Error('device lost');
await assert.rejects(scoreTargetSequence(pipeline, request), /device lost/);
assert.equal(resets, 3, 'reset after failed submitted scoring');
failure = null;
await assert.rejects(scoreTargetSequence({ ...pipeline,
  async prefillWithTargetTokenLogProbs(prompt, ids, options) { return { tokens: options.inputIds, targetTokenIds: ids, logProbs: [0.5, -1, -1] }; } },
request), /positive log-probability/);
await assert.rejects(scoreTargetSequence({ ...pipeline,
  async prefillWithTargetTokenLogProbs(prompt, ids, options) { return { tokens: options.inputIds, targetTokenIds: ids.slice(1), logProbs: [-1, -1] }; } },
request), /mismatched/);
assert.equal(resets, 5);

const controller = new AbortController();
during = () => controller.abort(new Error('cancel during submitted scoring'));
await assert.rejects(scoreTargetSequence(pipeline, request, { signal: controller.signal }), /cancel during submitted scoring/);
assert.equal(resets, 6);
const beforeAbort = prefills;
await assert.rejects(scoreTargetSequence(pipeline, request, { signal: controller.signal }), /cancel/);
assert.equal(prefills, beforeAbort, 'an aborted signal never reaches the GPU');
during = null;
assert.equal((await scoreTargetSequence(pipeline, request)).totalLogProb, -0.75, 'failed/cancelled scoring leaves reusable weights');

failure = new Error('submitted operation failed');
await assert.rejects(scoreTargetSequence({ ...pipeline, resetGenerationState() { throw new Error('cleanup failed'); } }, request),
  error => error instanceof AggregateError && error.errors[0] === failure && /cleanup failed/.test(error.errors[1].message));
failure = null;
await assert.rejects(scoreTargetSequence({ ...pipeline, resetGenerationState() { throw new Error('cleanup failed'); } }, request), /cleanup failed/);
await assert.rejects(scoreTargetSequence({ ...pipeline, prefillWithTargetTokenLogProbs: undefined }, request), /requires/);
console.log('target-sequence-scoring.test.js passed (injected operands and synthetic rows; no model-quality claim)');
