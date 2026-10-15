import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { requireComparable } from '../../benchmarks/vendors/generate-readme-evidence-svg.js';

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
