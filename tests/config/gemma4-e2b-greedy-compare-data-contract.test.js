import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const summary512Path = new URL('../../tools/data/gemma4-e2b-greedy-compare/summary-512.json', import.meta.url);
const interestingSummaryPath = new URL('../../tools/data/gemma4-e2b-greedy-compare/interesting-10-8tok-summary.json', import.meta.url);
const interestingPromptPackPath = new URL('../../tools/data/gemma4-e2b-greedy-compare/interesting-10-prompts.json', import.meta.url);

const summary512 = JSON.parse(await fs.readFile(summary512Path, 'utf8'));
const interestingSummary = JSON.parse(await fs.readFile(interestingSummaryPath, 'utf8'));
const interestingPromptPack = JSON.parse(await fs.readFile(interestingPromptPackPath, 'utf8'));

assert.deepEqual(summary512.aggregate, {
  promptCount: 512,
  sameFirstTokenCount: 329,
  sameFullTokenSequenceCount: 329,
  firstTokenMismatchCount: 183,
});
assert.equal(summary512.promptPackPath, 'tools/data/gemma4-e2b-blog-prompts-512.json');

assert.deepEqual(interestingSummary.aggregate, {
  promptCount: 10,
  sameFirstTokenCount: 0,
  sameFullTokenSequenceCount: 0,
  firstTokenMismatchCount: 10,
});
assert.equal(interestingSummary.promptPackPath, 'tools/data/gemma4-e2b-greedy-compare/interesting-10-prompts.json');
assert.equal(interestingPromptPack.length, 10);

const interestingIds = new Set(interestingSummary.prompts.map((entry) => entry.id));
for (const id of [
  'chem-equilibrium-stop-answer-oneword-d',
  'pol-voter-id-oneword-b',
  'tf-privacy-dead-oneword-a',
  'val-happiness-meaning-oneword-c',
  'ship-yes-no-both-oneword-d',
]) {
  assert.equal(interestingIds.has(id), true, `missing interesting compare prompt: ${id}`);
}

console.log('gemma4-e2b-greedy-compare-data-contract.test: ok');
