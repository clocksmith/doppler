import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  BLEU_SIGNATURE,
  CHRF_SIGNATURE,
  corpusBleu,
  corpusChrf,
  tokenize13a,
} from '../../tools/lib/sacrebleu-compatible-metrics.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const resultDir = path.join(
  root,
  'benchmarks/vendors/results/translategemma-nativekd2/quality-wmt13'
);
const stem = 'translategemma-nativekd2-wmt13-enes-128_20260711T012704Z';
const TOLERANCE = 1e-9;

function readJsonl(filename) {
  return fs.readFileSync(path.join(resultDir, filename), 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

// Golden check: recompute every group recorded by sacrebleu 2.6.0 in the
// retained receipt. Groups are rebuilt the way bench-translation-quality.js
// builds them: references are the trimmed dataset targets, source-bf16 uses the
// trimmed `pred`, engine lanes use `output`, and groups are overall plus each pair.
{
  const receipt = JSON.parse(
    fs.readFileSync(path.join(resultDir, `${stem}.receipt.json`), 'utf8')
  );
  const sourceRows = readJsonl(`${stem}.source-bf16.predictions.jsonl`);
  const datasetRows = sourceRows.map((row) => ({
    pair: row.pair.trim(),
    reference: row.target_pos.trim(),
  }));
  const sources = {
    'source-bf16': sourceRows.map((row) => row.pred.trim()),
  };
  for (const laneId of Object.keys(receipt.lanes)) {
    const laneRows = readJsonl(receipt.lanes[laneId].predictionsArtifact.filename);
    assert.equal(laneRows.length, datasetRows.length);
    laneRows.forEach((row, index) => {
      assert.equal(row.reference, datasetRows[index].reference, `${laneId} row ${index} reference`);
      assert.equal(row.pair, datasetRows[index].pair, `${laneId} row ${index} pair`);
    });
    sources[laneId] = laneRows.map((row) => row.output);
  }
  const pairs = [...new Set(datasetRows.map((row) => row.pair))].sort();
  const recorded = receipt.metrics.groups;
  const checked = new Set();
  const maxDiff = { bleu: 0, chrf: 0 };
  for (const [sourceId, predictions] of Object.entries(sources)) {
    const groupIndices = [['overall', datasetRows.map((_, index) => index)]];
    for (const pair of pairs) {
      groupIndices.push([
        pair,
        datasetRows.flatMap((row, index) => (row.pair === pair ? [index] : [])),
      ]);
    }
    for (const [groupName, indices] of groupIndices) {
      const groupId = `${sourceId}:${groupName}`;
      const expected = recorded[groupId];
      assert.ok(expected, `receipt is missing group ${groupId}`);
      const hypotheses = indices.map((index) => predictions[index]);
      const references = indices.map((index) => datasetRows[index].reference);
      assert.equal(expected.count, hypotheses.length, `${groupId} count`);
      const bleu = corpusBleu(hypotheses, references);
      const chrf = corpusChrf(hypotheses, references);
      const bleuDiff = Math.abs(bleu.score - expected.bleu);
      const chrfDiff = Math.abs(chrf.score - expected.chrf);
      assert.ok(bleuDiff <= TOLERANCE, `${groupId} BLEU ${bleu.score} vs ${expected.bleu}`);
      assert.ok(chrfDiff <= TOLERANCE, `${groupId} chrF ${chrf.score} vs ${expected.chrf}`);
      maxDiff.bleu = Math.max(maxDiff.bleu, bleuDiff);
      maxDiff.chrf = Math.max(maxDiff.chrf, chrfDiff);
      checked.add(groupId);
    }
  }
  assert.deepEqual([...checked].sort(), Object.keys(recorded).sort());
  console.log(
    `sacrebleu-compatible golden: ${checked.size} groups, `
    + `max |BLEU diff|=${maxDiff.bleu}, max |chrF diff|=${maxDiff.chrf}`
  );
}

// 13a tokenizer rules.
assert.equal(tokenize13a('Hello, world.'), 'Hello , world .');
assert.equal(tokenize13a('It costs 3,000.50 dollars'), 'It costs 3,000.50 dollars');
assert.equal(tokenize13a('pages 10-12'), 'pages 10 - 12');
assert.equal(tokenize13a('&quot;A&amp;B&quot; &lt;x&gt;'), '" A & B " < x >');
assert.equal(tokenize13a('a<skipped>b\nc-\nd'), 'ab cd');
assert.equal(tokenize13a('  spaced 　out  '), 'spaced out');

// Identical corpus scores 100 on both metrics.
{
  const hyp = ['the cat sat on the mat'];
  // exp(log(100)) is not exactly 100 in floating point, in sacrebleu as well.
  assert.ok(Math.abs(corpusBleu(hyp, hyp).score - 100) < 1e-12);
  assert.equal(corpusChrf(hyp, hyp).score, 100);
}

// Empty hypothesis: no matches, BLEU 0 with brevity penalty 0, chrF 0.
{
  const bleu = corpusBleu([''], ['a b c d']);
  assert.equal(bleu.score, 0);
  assert.equal(bleu.brevityPenalty, 0);
  assert.equal(bleu.sysLen, 0);
  assert.equal(bleu.refLen, 4);
  assert.equal(corpusChrf([''], ['abc']).score, 0);
}

// Zero 4-gram matches use exp smoothing: precisions 100, 100, 100, 100/(2*1).
{
  const bleu = corpusBleu(['a b c d'], ['a b c x d']);
  assert.deepEqual(bleu.counts, [4, 2, 1, 0]);
  assert.deepEqual(bleu.totals, [4, 3, 2, 1]);
  assert.deepEqual(bleu.precisions, [100, 200 / 3, 50, 50]);
  const brevityPenalty = Math.exp(1 - 5 / 4);
  const expected = brevityPenalty * Math.exp(
    (Math.log(100) + Math.log(200 / 3) + Math.log(50) + Math.log(50)) / 4
  );
  assert.ok(Math.abs(bleu.score - expected) < 1e-12);
}

// Hypothesis too short for 4-grams: total is 0 at order 4, so the loop breaks,
// that precision stays 0 and my_log(0) drives the score to 0.
{
  const bleu = corpusBleu(['a b c'], ['a b c']);
  assert.deepEqual(bleu.totals, [3, 2, 1, 0]);
  assert.equal(bleu.score, 0);
}

// chrF by hand: whitespace is removed and only orders present on both sides count.
// hyp "ab" vs ref "abc": order 1 P=1 R=2/3, order 2 P=1 R=1/2, orders 3-6 skipped.
{
  const chrf = corpusChrf(['a b'], ['abc']);
  const precision = 1;
  const recall = (2 / 3 + 1 / 2) / 2;
  const expected = 100 * (5 * precision * recall) / (4 * precision + recall);
  assert.ok(Math.abs(chrf.score - expected) < 1e-12);
}

// Code-point n-grams: an astral character is one character, as in Python.
{
  const chrf = corpusChrf(['\u{1F600}'], ['\u{1F600}x']);
  assert.deepEqual(chrf.statistics.slice(0, 3), [1, 2, 1]);
}

assert.match(BLEU_SIGNATURE, /^nrefs:1\|case:mixed\|eff:no\|tok:13a\|smooth:exp\|impl:doppler-js-sacrebleu-2\.6\.0-compatible$/u);
assert.match(CHRF_SIGNATURE, /^nrefs:1\|case:mixed\|eff:yes\|nc:6\|nw:0\|space:no\|impl:doppler-js-sacrebleu-2\.6\.0-compatible$/u);
assert.throws(() => corpusBleu(['a'], []), /1 hypotheses but 0 references/u);

console.log('sacrebleu-compatible-metrics.test: ok');
