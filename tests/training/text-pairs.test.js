import assert from 'node:assert/strict';

import {
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

assert.throws(
  () => normalizeTextPair({ prompt: 'A' }, 0),
  /requires completion\/target\/output/
);
assert.throws(
  () => normalizeTextPair({ prompt: 1, completion: 'B' }, 0),
  /field "prompt" for prompt\/source\/input must be a string/
);

console.log('text-pairs.test: ok');
