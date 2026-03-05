import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-training-workload-packs-'));
try {
  const outPath = path.join(tempDir, 'verify-output.json');
  const result = spawnSync(process.execPath, [
    'tools/verify-training-workload-packs.mjs',
    '--registry',
    'tools/configs/training-workloads/registry.json',
    '--out',
    outPath,
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });

  assert.equal(result.status, 0, result.stderr);
  const payload = JSON.parse(readFileSync(outPath, 'utf8'));
  assert.equal(payload.ok, true);
  assert.equal(payload.workloadCount >= 4, true);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('training-workload-packs.test: ok');
