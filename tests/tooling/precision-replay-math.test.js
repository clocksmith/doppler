import assert from 'node:assert/strict';
import fs from 'node:fs';

const {
  buildModeScoreMaps,
  compareTokenSequences,
  computeInversionCount,
  summarizeRanking,
} = await import('../../src/tooling/precision-replay-math.js');

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function createRowDecoder(rows) {
  const byId = new Map(rows.map((entry) => [entry.tokenId, entry.text]));
  return (tokenId) => byId.get(tokenId) ?? `[${tokenId}]`;
}

// === Real slice regression: backup yes/no watched pair flips under f16 ===

{
  const slice = readJson('reports/f16-precision-collapse/stable/curated/slices/backup-yes-no-choice.json');
  const hidden = Float32Array.from(slice.embedding);
  const candidateRows = new Map(
    slice.candidateRows.map((entry) => [entry.tokenId, Float32Array.from(entry.row)])
  );
  const scores = buildModeScoreMaps(hidden, candidateRows);
  const yesTokenId = slice.candidateRows.find((entry) => entry.text === ' yes')?.tokenId;
  const noTokenId = slice.candidateRows.find((entry) => entry.text === ' no')?.tokenId;

  assert.equal(typeof yesTokenId, 'number');
  assert.equal(typeof noTokenId, 'number');
  assert.ok(scores.exact.get(yesTokenId) > scores.exact.get(noTokenId));
  assert.ok(scores.f32_forward.get(yesTokenId) > scores.f32_forward.get(noTokenId));
  assert.ok(scores.f16_forward.get(yesTokenId) < scores.f16_forward.get(noTokenId));
}

// === Ranking + inversion helpers stay deterministic on a toy score set ===

{
  const tokenIds = [10, 20, 30];
  const leftScores = new Map([[10, 3], [20, 2], [30, 1]]);
  const rightScores = new Map([[10, 1], [20, 3], [30, 2]]);
  const ranking = summarizeRanking(tokenIds, leftScores, (tokenId) => `tok-${tokenId}`, 3);

  assert.equal(ranking.winnerTokenId, 10);
  assert.equal(ranking.winnerText, 'tok-10');
  assert.equal(ranking.winnerGap, 1);
  assert.deepEqual(
    ranking.top.map((entry) => entry.tokenId),
    [10, 20, 30]
  );
  assert.equal(computeInversionCount(tokenIds, leftScores, rightScores), 2);
}

// === Sequence comparison distinguishes persistence from healing ===

{
  const persistent = compareTokenSequences([1, 2, 3, 4], [1, 9, 8, 7]);
  assert.equal(persistent.firstDifferentStep, 1);
  assert.equal(persistent.healedAtStep, null);
  assert.equal(persistent.persistsThroughEnd, true);

  const healed = compareTokenSequences([1, 2, 3, 4], [1, 9, 3, 4]);
  assert.equal(healed.firstDifferentStep, 1);
  assert.equal(healed.healedAtStep, 2);
  assert.equal(healed.persistsThroughEnd, false);
}

// === Curated winner flip remains visible in a real slice ranking ===

{
  const slice = readJson('reports/f16-precision-collapse/stable/curated/slices/answer-is.json');
  const hidden = Float32Array.from(slice.embedding);
  const candidateRows = new Map(
    slice.candidateRows.map((entry) => [entry.tokenId, Float32Array.from(entry.row)])
  );
  const tokenIds = slice.candidateRows.map((entry) => entry.tokenId);
  const decode = createRowDecoder(slice.candidateRows);
  const scores = buildModeScoreMaps(hidden, candidateRows);
  const f32 = summarizeRanking(tokenIds, scores.f32_forward, decode, 8);
  const f16 = summarizeRanking(tokenIds, scores.f16_forward, decode, 8);

  assert.equal(f32.winnerText, ':');
  assert.equal(f16.winnerText, ' ');
  assert.notEqual(f32.winnerTokenId, f16.winnerTokenId);
}
