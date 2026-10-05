import assert from 'node:assert/strict';
import { scoreModelChoices } from '../../src/inference/choice-scoring.js';
import { snapshotChoiceScoringRequest, validateChoiceScoringResult } from '../../src/config/choice-scoring.js';

const request = { prompt: 'question:', choices: [{ id: 'accept', label: ' A' }, { id: 'reject', label: ' B' }], maxSeqLen: 8 };
let calls = 0, resets = 0, failure = null, during;
const pipeline = {
  tokenizer: { encode(text) {
    if (text === 'question:') return [1, 2];
    if (text === 'question: A') return [1, 2, 3];
    if (text === 'question: B') return [1, 2, 4];
    if (text === 'question: alias') return [1, 2, 3];
    if (text === 'question: merged') return [1, 9];
    return [1, 2, 3, 4];
  } },
  async prefillWithTokenLogits(prompt, ids, options) {
    calls++;
    assert.deepEqual(ids, [3, 4]);
    assert.deepEqual(options.inputIds, [1, 2]);
    assert.equal(options.useChatTemplate, false);
    during?.();
    if (failure) throw failure;
    return { tokens: [1, 2], logits: [1000, 1001] };
  },
  resetGenerationState() { resets++; },
};
const result = await scoreModelChoices(pipeline, request);
assert.equal(result.selectedId, 'reject');
assert.equal(result.calibration, null);
assert.equal(result.choices[0].logit, 1000);
assert.deepEqual(validateChoiceScoringResult(request, result), result);
assert.equal(resets, 1);
const snapshot = snapshotChoiceScoringRequest(request);
assert(Object.isFrozen(snapshot.choices[0]));
for (const bad of [
  { ...request, maxSeqLen: 1 }, { ...request, maxSeqLen: 0 },
  { ...request, choices: [{ id: 'a', label: ' A' }] },
  { ...request, choices: [{ id: 'a', label: ' A' }, { id: 'a', label: ' B' }] },
  { ...request, choices: [{ id: 'a', label: ' A' }, { id: 'b', label: ' alias' }] },
  { ...request, choices: [{ id: 'a', label: ' A' }, { id: 'b', label: ' merged' }] },
  { ...request, choices: [{ id: 'a', label: ' A' }, { id: 'b', label: ' multi token' }] },
  { ...request, useChatTemplate: true },
]) await assert.rejects(scoreModelChoices(pipeline, bad));
assert.equal(calls, 1, 'invalid labels and budgets fail before GPU execution');
for (const bad of [
  { ...result, selectedId: 'accept' }, { ...result, calibration: 'guaranteed' },
  { ...result, choices: result.choices.toReversed() },
  { ...result, choices: result.choices.map(choice => ({ ...choice, logit: NaN })) },
]) assert.throws(() => validateChoiceScoringResult(request, bad));
failure = new Error('device lost');
await assert.rejects(scoreModelChoices(pipeline, request), /device lost/);
assert.equal(resets, 2);
failure = null;
const controller = new AbortController();
during = () => controller.abort(new Error('cancel during submitted scoring'));
await assert.rejects(scoreModelChoices(pipeline, request, { signal: controller.signal }), /cancel during submitted scoring/);
assert.equal(resets, 3);
const before = calls;
await assert.rejects(scoreModelChoices(pipeline, request, { signal: controller.signal }), /cancel/);
assert.equal(calls, before);
during = null;
assert.equal((await scoreModelChoices(pipeline, request)).selectedId, 'reject', 'failed/cancelled scoring leaves reusable weights');
failure = new Error('submitted operation failed');
await assert.rejects(scoreModelChoices({ ...pipeline, resetGenerationState() { throw new Error('cleanup failed'); } }, request),
  error => error instanceof AggregateError && error.errors[0] === failure && /cleanup failed/.test(error.errors[1].message));
failure = null;
await assert.rejects(scoreModelChoices({ ...pipeline, resetGenerationState() { throw new Error('cleanup failed'); } }, request), /cleanup failed/);
await assert.rejects(scoreModelChoices({ ...pipeline, resetGenerationState: undefined }, request), /requires/);
console.log('choice-scoring.test.js passed (injected operands; no model-quality claim)');
