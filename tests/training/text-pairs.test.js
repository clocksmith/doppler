import assert from 'node:assert/strict';

import {
  CAUSAL_LM_IGNORE_TARGET_ID,
  loadTextPairsDataset,
  mapTextPairs,
  normalizeTextPair,
  parseTextPairsDataset,
  tokenizeTextPairs,
} from '../../src/experimental/training/datasets/text-pairs.js';

const tokenizer = {
  encode(text) {
    return Array.from(text).map((character) => character.codePointAt(0));
  },
};

const columboRow = {
  rowId: 'columbo-row-1',
  source: 'system: Return JSON.\n\nuser: {"input":{"text":"alpha"}}',
  target: '{"findings":[]}',
};

assert.deepEqual(normalizeTextPair(columboRow, 0), {
  id: 'columbo-row-1',
  prompt: columboRow.source,
  completion: columboRow.target,
  promptField: 'source',
  completionField: 'target',
});

assert.deepEqual(mapTextPairs([
  { id: 'prompt-row', prompt: 'A', completion: 'B' },
  { rowId: 'io-row', input: 'C', output: 'D' },
]).map((row) => ({
  id: row.id,
  prompt: row.prompt,
  completion: row.completion,
  promptField: row.promptField,
  completionField: row.completionField,
})), [
  { id: 'prompt-row', prompt: 'A', completion: 'B', promptField: 'prompt', completionField: 'completion' },
  { id: 'io-row', prompt: 'C', completion: 'D', promptField: 'input', completionField: 'output' },
]);

const datasetText = [
  JSON.stringify(columboRow),
  JSON.stringify({ id: 'second-row', prompt: 'Second', completion: 'Done' }),
].join('\n');
assert.deepEqual(parseTextPairsDataset(datasetText, { sourceLabel: 'rows.jsonl' }).rows.map((row) => row.id), [
  'columbo-row-1',
  'second-row',
]);

const loaded = await loadTextPairsDataset('/tmp/doppler-text-pairs.jsonl', {
  readFile: async () => datasetText,
});
assert.equal(loaded.rowCount, 2);
assert.equal(loaded.rows[0].promptField, 'source');

const samples = await tokenizeTextPairs(tokenizer, [columboRow], {
  joinWith: '\nassistant: ',
  maxLength: 512,
});
assert.equal(samples.length, 1);
assert.equal(samples[0].id, 'columbo-row-1');
assert.equal(samples[0].prompt, columboRow.source);
assert.equal(samples[0].completion, columboRow.target);
assert.equal(samples[0].text, `${columboRow.source}\nassistant: ${columboRow.target}`);
assert.equal(samples[0].inputIds.length, samples[0].targetIds.length);
assert.equal(samples[0].targetIds.at(-1), '}'.codePointAt(0));
assert.equal(samples[0].ignoredTargetCount, `${columboRow.source}\nassistant: `.length - 1);
assert.equal(samples[0].supervisedTokenCount, columboRow.target.length);
assert.equal(samples[0].completionTokenCount, columboRow.target.length);
assert.equal(samples[0].truncatedPromptTokenCount, 0);
assert.ok(
  samples[0].targetIds
    .slice(0, samples[0].ignoredTargetCount)
    .every((token) => token === CAUSAL_LM_IGNORE_TARGET_ID)
);
assert.deepEqual(
  samples[0].targetIds.slice(samples[0].ignoredTargetCount),
  tokenizer.encode(columboRow.target)
);

const [truncated] = await tokenizeTextPairs(tokenizer, [{
  id: 'truncated-prompt',
  prompt: 'ABCDEFGHIJ',
  completion: 'xyz',
}], {
  maxLength: 7,
});
assert.deepEqual(truncated.inputIds, [
  'A'.codePointAt(0),
  'B'.codePointAt(0),
  'I'.codePointAt(0),
  'J'.codePointAt(0),
  'x'.codePointAt(0),
  'y'.codePointAt(0),
]);
assert.deepEqual(truncated.targetIds, [
  CAUSAL_LM_IGNORE_TARGET_ID,
  CAUSAL_LM_IGNORE_TARGET_ID,
  CAUSAL_LM_IGNORE_TARGET_ID,
  'x'.codePointAt(0),
  'y'.codePointAt(0),
  'z'.codePointAt(0),
]);
assert.equal(truncated.promptTokenCount, 10);
assert.equal(truncated.retainedPromptTokenCount, 4);
assert.equal(truncated.truncatedPromptTokenCount, 6);
assert.equal(truncated.supervisedTokenCount, 3);

const bosTokenizer = {
  encode(text) {
    return [2, ...tokenizer.encode(text)];
  },
};
const [bosSample] = await tokenizeTextPairs(bosTokenizer, [{
  id: 'single-bos',
  prompt: 'AB',
  completion: 'xyz',
}], {
  maxLength: 6,
});
assert.deepEqual(bosSample.inputIds, [
  2,
  'A'.codePointAt(0),
  'B'.codePointAt(0),
  'x'.codePointAt(0),
  'y'.codePointAt(0),
]);
assert.equal(bosSample.inputIds.filter((token) => token === 2).length, 1);
assert.equal(bosSample.completionTokenCount, 3);

await assert.rejects(
  tokenizeTextPairs(tokenizer, [{
    id: 'oversized-completion',
    prompt: 'A',
    completion: 'xyz',
  }], {
    maxLength: 3,
  }),
  /must also retain at least one prompt token/
);

assert.throws(
  () => normalizeTextPair({ prompt: 'A' }, 0),
  /requires completion\/target\/output/
);
assert.throws(
  () => normalizeTextPair({ prompt: 1, completion: 'B' }, 0),
  /field "prompt" for prompt\/source\/input must be a string/
);

console.log('text-pairs.test: ok');
