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

function normalizeCatalogTestedState(value) {
  const text = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (text === 'pass') return 'verified';
  return text || 'unknown';
}

async function readExpectedReleaseClaimableModelIds() {
  const catalogPath = path.join(process.cwd(), 'models', 'catalog.json');
  const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
  return (Array.isArray(catalog?.models) ? catalog.models : [])
    .filter((entry) => {
      if (!entry || typeof entry !== 'object') return false;
      const lifecycle = entry.lifecycle && typeof entry.lifecycle === 'object' ? entry.lifecycle : {};
      const status = lifecycle.status && typeof lifecycle.status === 'object' ? lifecycle.status : {};
      const tested = lifecycle.tested && typeof lifecycle.tested === 'object' ? lifecycle.tested : {};
      const runtimeStatus = typeof status.runtime === 'string' ? status.runtime.trim().toLowerCase() : 'unknown';
      const testedStatus = normalizeCatalogTestedState(tested.result ?? status.tested);
      return runtimeStatus === 'active' && testedStatus === 'verified';
    })
    .map((entry) => entry.modelId)
    .sort();
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
  const gitStatus = spawnSync('git', ['status', '--porcelain'], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  const expectedDirty = gitStatus.status === 0
    ? gitStatus.stdout.trim().length > 0
    : null;
  const expectedModelCoverage = await readExpectedReleaseClaimableModelIds();
  assert.equal(matrixPayload.generatedAt, timestamp);
  assert.equal(matrixPayload.release?.dirty, expectedDirty);
  assert.ok(matrixPayload.sources?.compareMetricContract);
  assert.ok(matrixPayload.sources?.benchmarkPolicy);
  assert.ok(matrixPayload.sources?.['harness:doppler']);
  assert.ok(matrixPayload.sources?.['harness:transformersjs']);
  assert.deepEqual(
    matrixPayload.modelCoverage.map((entry) => entry.dopplerModelId).sort(),
    expectedModelCoverage
  );
  assert.match(markdownPayload, /^# Release Matrix/m);
  assert.match(markdownPayload, /Generated: 2026-03-05T00:00:00.000Z/);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-vendor-bench-compare-'));
  const fixturePaths = [
    path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-1b-p064-d064-t0-k1.compare.json'),
    path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-p064-d064-t0-k1.apple-m3pro.compare.json'),
    path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-p064-d064-t0-k1.compare.json'),
    path.join(process.cwd(), 'benchmarks', 'vendors', 'fixtures', 'g3-p064-d064-t1-k32.compare.json'),
  ];
  for (const fixturePath of fixturePaths) {
    const fixturePayload = JSON.parse(await fs.readFile(fixturePath, 'utf8'));
    assert.equal(fixturePayload.benchmarkPolicy?.source, 'benchmarks/vendors/benchmark-policy.json');
  }

  const copiedFixturePath = path.join(tempDir, 'stale.compare.json');
  const fixturePayload = JSON.parse(await fs.readFile(fixturePaths[0], 'utf8'));
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
  assert.match(result.stderr, /stale (benchmarkPolicy|compareConfig|metricContract|dopplerHarness|transformersjsHarness) hash/);
}

console.log('vendor-bench-cli-contract.test: ok');
