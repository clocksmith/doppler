import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function runVendorBench(args) {
  return spawnSync(process.execPath, ['tools/vendor-bench.js', ...args], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
}

{
  const result = runVendorBench(['list']);
  assert.equal(result.status, 0, result.stderr);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-vendor-bench-'));
  const matrixPath = path.join(tempDir, 'release-matrix.json');
  const markdownPath = path.join(tempDir, 'release-matrix.md');
  const timestamp = '2026-03-05T00:00:00.000Z';
  const result = runVendorBench([
    'matrix',
    '--timestamp', timestamp,
    '--output', matrixPath,
    '--markdown-output', markdownPath,
  ]);
  assert.equal(result.status, 0, result.stderr);

  const matrixPayload = JSON.parse(await fs.readFile(matrixPath, 'utf8'));
  const markdownPayload = await fs.readFile(markdownPath, 'utf8');
  assert.equal(matrixPayload.generatedAt, timestamp);
  assert.equal(matrixPayload.release?.dirty, null);
  assert.ok(matrixPayload.sources?.compareMetricContract);
  assert.ok(matrixPayload.sources?.benchmarkPolicy);
  assert.ok(matrixPayload.sources?.['harness:doppler']);
  assert.ok(matrixPayload.sources?.['harness:transformersjs']);
  assert.match(markdownPayload, /^# Release Matrix/m);
  assert.match(markdownPayload, /Generated: 2026-03-05T00:00:00.000Z/);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-vendor-bench-compare-'));
  const fixturePath = path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-1b-p064-d064-t0-k1.compare.json');
  const copiedFixturePath = path.join(tempDir, 'stale.compare.json');
  const fixturePayload = JSON.parse(await fs.readFile(fixturePath, 'utf8'));
  fixturePayload.metricContract.sourceSha256 = 'stale-hash';
  await fs.writeFile(copiedFixturePath, `${JSON.stringify(fixturePayload, null, 2)}\n`, 'utf8');

  const matrixPath = path.join(tempDir, 'release-matrix.json');
  const markdownPath = path.join(tempDir, 'release-matrix.md');
  const result = runVendorBench([
    'matrix',
    '--compare-result', copiedFixturePath,
    '--output', matrixPath,
    '--markdown-output', markdownPath,
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /stale (compareConfig|metricContract|dopplerHarness|transformersjsHarness) hash/);
}

console.log('vendor-bench-cli-contract.test: ok');
