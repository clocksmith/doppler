import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { loadHighlight, requireComparable } from '../../benchmarks/vendors/generate-readme-evidence-svg.js';

const catalog = JSON.parse(await readFile('models/catalog.json', 'utf8'));
const embeddingPath = 'benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260710T011455.json';
const embedding = JSON.parse(await readFile(embeddingPath, 'utf8'));
const highlight = { kind: 'embedding', receipt: embeddingPath };
loadHighlight(highlight, embedding, catalog);
for (const reject of [
  (copy) => { copy.model.transformersjsModelId = 'superseded-comparator'; },
  (copy) => { copy.model.transformersjsDtype = 'fp32'; },
  (copy) => { copy.fairness.releaseClaimable = false; },
  (copy) => { copy.raw.transformersjsBench.deviceInfo.device = 'different-gpu'; },
]) {
  const copy = structuredClone(embedding);
  reject(copy);
  assert.throws(() => loadHighlight(highlight, copy, catalog), /gate failed/);
}

const receiptPath = 'benchmarks/vendors/results/compare_20260707T153509.json';
const receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
const definition = { kind: 'text', section: 'parity', receipt: receiptPath };
requireComparable(receipt, definition);
for (const reject of [
  (copy) => { copy.fairness.claimGrade = false; },
  (copy) => { copy.fairness.correctnessOk = false; },
  (copy) => { copy.fairness.surfaceFairness.sameRuntimeSurface = false; },
  (copy) => { copy.fairness.sections['compute/parity'].claimGrade = false; },
  (copy) => { copy.sections.compute.parity.decodeValidity.ok = false; },
  (copy) => { copy.sections.compute.parity.pairedComparable = false; },
]) {
  const copy = structuredClone(receipt);
  reject(copy);
  assert.throws(() => requireComparable(copy, definition), /gate failed/);
}

const paired = JSON.parse(await readFile(
  'benchmarks/vendors/results/qwen35-2b-metal-paired-p256-d512-20260710.json', 'utf8'
));
requireComparable(paired, { kind: 'paired-text', receipt: 'paired fixture' });
paired.correctness.exactTokenParityAllPairs = false;
assert.throws(() => requireComparable(paired, { kind: 'paired-text', receipt: 'paired fixture' }), /gate failed/);

const result = spawnSync(
  process.execPath,
  ['tools/render-readme-evidence-chart.js', '--check'],
  {
    cwd: process.cwd(),
    encoding: 'utf8',
  }
);

assert.equal(
  result.status,
  0,
  `README evidence chart check failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`
);

const svg = await readFile('assets/doppler-webgpu-evidence.svg', 'utf8');
const benchmarkRows = svg.match(/class="benchmark-row"/g) ?? [];
const receiptLinks = svg.match(/data-receipt="benchmarks\/vendors\/results\//g) ?? [];

assert.ok(benchmarkRows.length > 0, 'README evidence chart must contain comparable rows');
assert.equal(receiptLinks.length, benchmarkRows.length, 'every chart row must bind its source receipt');

console.log('readme-evidence-chart-contract.test: ok');
