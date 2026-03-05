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
  assert.match(markdownPayload, /^# Release Matrix/m);
  assert.match(markdownPayload, /Generated: 2026-03-05T00:00:00.000Z/);
}

console.log('vendor-bench-cli-contract.test: ok');
