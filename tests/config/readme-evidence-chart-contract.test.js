import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';

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
