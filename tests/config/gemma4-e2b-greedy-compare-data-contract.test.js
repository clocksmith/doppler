import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const summaryPaths = [
  'tools/data/gemma4-e2b-greedy-compare/summary-512.json',
  'tools/data/gemma4-e2b-greedy-compare/interesting-10-8tok-summary.json',
];

async function readJson(relativePath) {
  return JSON.parse(await fs.readFile(new URL(`../../${relativePath}`, import.meta.url), 'utf8'));
}

for (const summaryPath of summaryPaths) {
  const summary = await readJson(summaryPath);
  const aggregate = summary.aggregate;
  // Original benchmark record fields are retained verbatim, not runtime aliases.
  const promptPack = await readJson(summary.promptPackPath);

  assert.equal(aggregate.promptCount, summary.prompts.length, `${summaryPath}: promptCount must match recorded prompts`);
  assert.equal(aggregate.promptCount, promptPack.length, `${summaryPath}: promptCount must match the referenced prompt pack`);
  assert.equal(
    aggregate.sameFirstTokenCount + aggregate.firstTokenMismatchCount,
    aggregate.promptCount,
    `${summaryPath}: first-token matches and mismatches must partition the prompts`
  );
  assert.ok(
    aggregate.sameFullTokenSequenceCount <= aggregate.sameFirstTokenCount,
    `${summaryPath}: full-sequence matches cannot exceed first-token matches`
  );
}

console.log('gemma4-e2b-greedy-compare-data-contract.test: ok');
