import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

const result = spawnSync(process.execPath, [
  'tools/verify-training-workload-packs.mjs',
  '--registry',
  'tools/configs/training-workloads/registry.json',
], {
  cwd: process.cwd(),
  encoding: 'utf8',
});

assert.equal(result.status, 0, result.stderr);
const payload = JSON.parse(result.stdout);
assert.equal(payload.ok, true);
assert.equal(payload.workloadCount >= 4, true);

console.log('training-workload-packs.test: ok');
