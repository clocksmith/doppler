import assert from 'node:assert/strict';

import { SpeculativeDecoder } from '../../src/inference/speculative.js';

function makeConfig(overrides = {}) {
  return {
    numDraftTokens: 4,
    maxRejectionRetries: 3,
    enableTreeDraft: false,
    temperature: 1.0,
    randomSeed: 42,
    ...overrides,
  };
}

function makeLogits(vocabSize, hotIndex) {
  const logits = new Float32Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    logits[i] = -10;
  }
  logits[hotIndex] = 10;
  return logits;
}

function makeMockModel(vocabSize, tokenSequence) {
  let callIndex = 0;
  return {
    forward(inputIds, kvCache) {
      const token = tokenSequence[callIndex % tokenSequence.length];
      callIndex++;
      const logits = makeLogits(vocabSize, token);
      return { logits, newKVCache: kvCache };
    },
  };
}

// === Constructor validation ===

{
  assert.throws(
    () => new SpeculativeDecoder({}),
    /requires numDraftTokens/
  );
  assert.throws(
    () => new SpeculativeDecoder({ numDraftTokens: 4 }),
    /requires maxRejectionRetries/
  );
  assert.throws(
    () => new SpeculativeDecoder({ numDraftTokens: 4, maxRejectionRetries: 3 }),
    /requires enableTreeDraft/
  );
  assert.throws(
    () => new SpeculativeDecoder({
      numDraftTokens: 4,
      maxRejectionRetries: 3,
      enableTreeDraft: false,
    }),
    /requires temperature/
  );
  assert.throws(
    () => new SpeculativeDecoder({
      numDraftTokens: 4,
      maxRejectionRetries: 3,
      enableTreeDraft: false,
      temperature: 1.0,
    }),
    /requires randomSeed/
  );
}

// === Constructor rejects invalid temperature ===

{
  assert.throws(
    () => new SpeculativeDecoder(makeConfig({ temperature: 0 })),
    /positive finite number/
  );
  assert.throws(
    () => new SpeculativeDecoder(makeConfig({ temperature: -1 })),
    /positive finite number/
  );
  assert.throws(
    () => new SpeculativeDecoder(makeConfig({ temperature: Infinity })),
    /positive finite number/
  );
  assert.throws(
    () => new SpeculativeDecoder(makeConfig({ temperature: NaN })),
    /positive finite number/
  );
}

// === Valid construction ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  assert.equal(decoder.numDraftTokens, 4);
  assert.equal(decoder.maxRejectionRetries, 3);
  assert.equal(decoder.enableTreeDraft, false);
  assert.equal(decoder.temperature, 1.0);
  assert.equal(decoder.draftModel, null);
  assert.equal(decoder.mainModel, null);
}

// === setDraftModel / setMainModel ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  const draft = { forward: () => ({}) };
  const main = { forward: () => ({}) };
  decoder.setDraftModel(draft);
  decoder.setMainModel(main);
  assert.equal(decoder.draftModel, draft);
  assert.equal(decoder.mainModel, main);
}

// === logSoftmax produces valid log-probabilities ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  const logits = new Float32Array([1.0, 2.0, 3.0, 4.0]);
  const result = decoder.logSoftmax(logits);

  assert.equal(result.length, 4);
  for (let i = 0; i < result.length; i++) {
    assert.ok(result[i] <= 0, 'log-probabilities must be non-positive');
  }

  let sumProbs = 0;
  for (let i = 0; i < result.length; i++) {
    sumProbs += Math.exp(result[i]);
  }
  assert.ok(Math.abs(sumProbs - 1.0) < 1e-5, `sum of softmax probabilities should be ~1, got ${sumProbs}`);
}

// === logSoftmax is numerically stable with large values ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  const logits = new Float32Array([1000, 1001, 999]);
  const result = decoder.logSoftmax(logits);

  for (let i = 0; i < result.length; i++) {
    assert.ok(Number.isFinite(result[i]), `logSoftmax result[${i}] should be finite`);
  }

  let sumProbs = 0;
  for (let i = 0; i < result.length; i++) {
    sumProbs += Math.exp(result[i]);
  }
  assert.ok(Math.abs(sumProbs - 1.0) < 1e-5);
}

// === logSoftmax preserves ordering ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  const logits = new Float32Array([1.0, 5.0, 3.0]);
  const result = decoder.logSoftmax(logits);
  assert.ok(result[1] > result[2]);
  assert.ok(result[2] > result[0]);
}

// === sampleToken returns valid token index and logprobs ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  const logits = makeLogits(8, 3);
  const { token, logprob } = decoder.sampleToken(logits, 1.0);

  assert.ok(Number.isInteger(token));
  assert.ok(token >= 0 && token < 8);
  assert.ok(logprob instanceof Float32Array);
  assert.equal(logprob.length, 8);
  assert.equal(token, 3, 'should sample the heavily favored token');
}

// === sampleToken with low temperature concentrates mass ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  const logits = new Float32Array([0, 0, 0, 5, 0, 0]);
  const { token } = decoder.sampleToken(logits, 0.01);
  assert.equal(token, 3, 'low temperature should always pick the top token');
}

// === sampleToken rejects invalid temperature ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  const logits = new Float32Array([1.0, 2.0]);
  assert.throws(() => decoder.sampleToken(logits, 0), /positive finite/);
  assert.throws(() => decoder.sampleToken(logits, -1), /positive finite/);
  assert.throws(() => decoder.sampleToken(logits, NaN), /positive finite/);
}

// === sampleToken rejects empty logits ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  assert.throws(
    () => decoder.sampleToken(new Float32Array(0), 1.0),
    /must not be empty/
  );
  assert.throws(
    () => decoder.sampleToken([], 1.0),
    /must be a numeric logits vector/
  );
}

// === coerceLogitsVector via sampleToken accepts various input shapes ===

{
  const decoder = new SpeculativeDecoder(makeConfig());

  const fromArray = decoder.sampleToken([0, 0, 10, 0], 1.0);
  assert.equal(fromArray.token, 2);

  const fromFloat32 = decoder.sampleToken(new Float32Array([0, 0, 0, 10]), 1.0);
  assert.equal(fromFloat32.token, 3);

  const nestedLast = decoder.sampleToken([new Float32Array([0, 10, 0])], 1.0);
  assert.equal(nestedLast.token, 1);

  const nestedArrayLast = decoder.sampleToken([[10, 0, 0]], 1.0);
  assert.equal(nestedArrayLast.token, 0);
}

// === generateDraftTokens requires draft model ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  await assert.rejects(
    () => decoder.generateDraftTokens([1], null),
    /Draft model not set/
  );
}

// === generateDraftTokens produces correct number of tokens ===

{
  const vocabSize = 16;
  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: 3 }));
  decoder.setDraftModel(makeMockModel(vocabSize, [5, 7, 2]));

  const result = await decoder.generateDraftTokens([0], null, 3);
  assert.equal(result.tokens.length, 3);
  assert.equal(result.logprobs.length, 3);

  for (const tok of result.tokens) {
    assert.ok(Number.isInteger(tok));
    assert.ok(tok >= 0 && tok < vocabSize);
  }
}

// === verifyDraftTokens requires main model ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  await assert.rejects(
    () => decoder.verifyDraftTokens([1], [2], [new Float32Array([0.5])], null),
    /Main model not set/
  );
}

// === verifyDraftTokens rejects mismatched token/logprob arrays ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  decoder.setMainModel(makeMockModel(4, [0]));

  await assert.rejects(
    () => decoder.verifyDraftTokens([0], [1, 2], [new Float32Array(4)], null),
    /length mismatch/
  );
}

// === verifyDraftTokens rejects non-array arguments ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  decoder.setMainModel(makeMockModel(4, [0]));

  await assert.rejects(
    () => decoder.verifyDraftTokens([0], 'not-an-array', [], null),
    /must be arrays/
  );
}

// === verifyDraftTokens: all accepted when draft == main distribution ===

{
  const vocabSize = 8;
  const targetToken = 3;
  const numDraft = 3;

  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: numDraft }));
  decoder.setDraftModel(makeMockModel(vocabSize, [targetToken]));
  decoder.setMainModel(makeMockModel(vocabSize, [targetToken]));

  const draft = await decoder.generateDraftTokens([0], null, numDraft);

  assert.equal(draft.tokens.length, numDraft);
  for (const tok of draft.tokens) {
    assert.equal(tok, targetToken);
  }

  const result = await decoder.verifyDraftTokens([0], draft.tokens, draft.logprobs, null);

  assert.equal(result.allAccepted, true);
  assert.equal(result.acceptedCount, numDraft);
  assert.equal(result.acceptedTokens.length, numDraft);
  assert.ok(Number.isInteger(result.sampledToken));
  assert.ok(result.sampledToken >= 0 && result.sampledToken < vocabSize);
}

// === verifyDraftTokens: rejection when distributions disagree ===

{
  const vocabSize = 8;
  const decoder = new SpeculativeDecoder(makeConfig({
    numDraftTokens: 2,
    temperature: 1.0,
    randomSeed: 7,
  }));

  const draftToken = 1;
  const mainToken = 5;

  decoder.setDraftModel(makeMockModel(vocabSize, [draftToken]));
  decoder.setMainModel(makeMockModel(vocabSize, [mainToken]));

  const draft = await decoder.generateDraftTokens([0], null, 2);
  assert.equal(draft.tokens[0], draftToken);

  const result = await decoder.verifyDraftTokens([0], draft.tokens, draft.logprobs, null);

  assert.ok(result.acceptedCount < 2, 'should reject at least one token when distributions strongly disagree');
  assert.ok(Number.isInteger(result.sampledToken));
}

// === verifyDraftTokens rejects out-of-range draft token ===

{
  const vocabSize = 4;
  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: 1 }));
  decoder.setMainModel(makeMockModel(vocabSize, [0]));

  const draftLogprobs = [new Float32Array(vocabSize).fill(-1)];
  await assert.rejects(
    () => decoder.verifyDraftTokens([0], [vocabSize], draftLogprobs, null),
    /out of vocabulary range/
  );

  await assert.rejects(
    () => decoder.verifyDraftTokens([0], [-1], draftLogprobs, null),
    /out of vocabulary range/
  );
}

// === verifyDraftTokens rejects vocab size mismatch between draft and main ===

{
  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: 1 }));
  decoder.setMainModel(makeMockModel(8, [0]));

  const mismatchedLogprobs = [new Float32Array(4).fill(-1)];
  await assert.rejects(
    () => decoder.verifyDraftTokens([0], [0], mismatchedLogprobs, null),
    /does not match main logits length/
  );
}

// === stats tracking ===

{
  const vocabSize = 8;
  const targetToken = 2;
  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: 2 }));
  decoder.setDraftModel(makeMockModel(vocabSize, [targetToken]));
  decoder.setMainModel(makeMockModel(vocabSize, [targetToken]));

  const initialStats = decoder.getStats();
  assert.equal(initialStats.totalDrafted, 0);
  assert.equal(initialStats.totalAccepted, 0);
  assert.equal(initialStats.totalRejected, 0);
  assert.equal(initialStats.averageAcceptRate, 0);

  const draft = await decoder.generateDraftTokens([0], null, 2);
  await decoder.verifyDraftTokens([0], draft.tokens, draft.logprobs, null);

  const afterStats = decoder.getStats();
  assert.equal(afterStats.totalDrafted, 2);
  assert.ok(afterStats.totalAccepted >= 0);
  assert.ok(afterStats.totalRejected >= 0);
  assert.equal(afterStats.totalAccepted + afterStats.totalRejected, 2);
  assert.ok(Number.isFinite(afterStats.averageAcceptRate));
  assert.ok(Number.isFinite(afterStats.speedup));
}

// === resetStats clears counters ===

{
  const vocabSize = 8;
  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: 2 }));
  decoder.setDraftModel(makeMockModel(vocabSize, [0]));
  decoder.setMainModel(makeMockModel(vocabSize, [0]));

  const draft = await decoder.generateDraftTokens([0], null, 2);
  await decoder.verifyDraftTokens([0], draft.tokens, draft.logprobs, null);
  assert.ok(decoder.stats.totalDrafted > 0);

  decoder.resetStats();
  assert.equal(decoder.stats.totalDrafted, 0);
  assert.equal(decoder.stats.totalAccepted, 0);
  assert.equal(decoder.stats.totalRejected, 0);
  assert.equal(decoder.stats.averageAcceptRate, 0);
}

// === estimateSpeedup returns 1.0 when no tokens drafted ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  assert.equal(decoder.estimateSpeedup(), 1.0);
}

// === estimateSpeedup reflects accept rate ===

{
  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: 4 }));
  decoder.stats.totalDrafted = 100;
  decoder.stats.totalAccepted = 80;
  decoder.stats.totalRejected = 20;
  decoder.stats.averageAcceptRate = 0.8;

  const speedup = decoder.estimateSpeedup();
  assert.ok(speedup > 1.0, `speedup with 80% accept rate should be > 1.0, got ${speedup}`);
}

// === sampleFromResidual without rejection delegates to sampleToken ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  const mainLogits = makeLogits(8, 5);
  const draftLogprobs = new Float32Array(8).fill(-2);
  const token = decoder.sampleFromResidual(mainLogits, draftLogprobs, false);
  assert.ok(Number.isInteger(token));
  assert.ok(token >= 0 && token < 8);
  assert.equal(token, 5, 'without rejection, should sample from main logits directly');
}

// === sampleFromResidual with rejection uses residual distribution ===

{
  const vocabSize = 4;
  const decoder = new SpeculativeDecoder(makeConfig({ randomSeed: 99 }));

  const mainLogits = new Float32Array([10, -10, -10, -10]);
  const draftLogprobs = new Float32Array(vocabSize);
  const draftLogprobValues = decoder.logSoftmax(new Float32Array([-10, 10, -10, -10]));
  draftLogprobs.set(draftLogprobValues);

  const token = decoder.sampleFromResidual(mainLogits, draftLogprobs, true);
  assert.ok(Number.isInteger(token));
  assert.ok(token >= 0 && token < vocabSize);
  assert.equal(token, 0, 'residual should favor token 0 (high main prob, low draft prob)');
}

// === sampleFromResidual rejects length mismatch ===

{
  const decoder = new SpeculativeDecoder(makeConfig());
  assert.throws(
    () => decoder.sampleFromResidual(new Float32Array(4), new Float32Array(8), true),
    /length mismatch/
  );
}

// === step end-to-end: produces newTokens ===

{
  const vocabSize = 8;
  const targetToken = 4;
  const numDraft = 3;

  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: numDraft }));
  decoder.setDraftModel(makeMockModel(vocabSize, [targetToken]));
  decoder.setMainModel(makeMockModel(vocabSize, [targetToken]));

  const result = await decoder.step([0], null, null);

  assert.ok(Array.isArray(result.newTokens));
  assert.ok(result.newTokens.length >= 1, 'step must produce at least one token');
  assert.ok(result.newTokens.length <= numDraft + 1, 'step cannot produce more than numDraft+1 tokens');
  assert.ok(Number.isFinite(result.acceptRate));
  assert.ok(result.acceptRate >= 0 && result.acceptRate <= 1);
}

// === step with matching models accepts all drafts ===

{
  const vocabSize = 8;
  const targetToken = 6;
  const numDraft = 2;

  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: numDraft }));
  decoder.setDraftModel(makeMockModel(vocabSize, [targetToken]));
  decoder.setMainModel(makeMockModel(vocabSize, [targetToken]));

  const result = await decoder.step([0], null, null);

  assert.equal(result.newTokens.length, numDraft + 1, 'all drafts accepted + one sampled continuation');
  assert.equal(result.acceptRate, 1.0);
}

// === RNG determinism: same seed produces same sequence ===

{
  const d1 = new SpeculativeDecoder(makeConfig({ randomSeed: 123 }));
  const d2 = new SpeculativeDecoder(makeConfig({ randomSeed: 123 }));

  const logits = new Float32Array([1, 2, 3, 4, 5]);
  const results1 = [];
  const results2 = [];

  for (let i = 0; i < 10; i++) {
    results1.push(d1.sampleToken(logits, 1.0).token);
    results2.push(d2.sampleToken(logits, 1.0).token);
  }

  assert.deepEqual(results1, results2, 'same seed must produce identical token sequences');
}

// === RNG determinism: different seeds produce different sequences ===

{
  const d1 = new SpeculativeDecoder(makeConfig({ randomSeed: 1 }));
  const d2 = new SpeculativeDecoder(makeConfig({ randomSeed: 9999 }));

  const logits = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]);
  const results1 = [];
  const results2 = [];

  for (let i = 0; i < 20; i++) {
    results1.push(d1.sampleToken(logits, 1.0).token);
    results2.push(d2.sampleToken(logits, 1.0).token);
  }

  let anyDiff = false;
  for (let i = 0; i < results1.length; i++) {
    if (results1[i] !== results2[i]) {
      anyDiff = true;
      break;
    }
  }
  assert.ok(anyDiff, 'different seeds should eventually produce different samples');
}

// === kvCache clone is called when available ===

{
  const vocabSize = 4;
  let cloneCalled = 0;
  const kvCache = {
    clone() {
      cloneCalled++;
      return { clone() { cloneCalled++; return this; } };
    },
  };

  const decoder = new SpeculativeDecoder(makeConfig({ numDraftTokens: 1 }));
  decoder.setDraftModel(makeMockModel(vocabSize, [0]));

  await decoder.generateDraftTokens([0], kvCache, 1);
  assert.ok(cloneCalled >= 1, 'kvCache.clone should be called during draft generation');
}

console.log('speculative-decoding.test: ok');
