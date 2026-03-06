import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

const result = spawnSync(
  process.execPath,
  ['tools/check-contract-artifacts.js', '--json'],
  {
    cwd: process.cwd(),
    encoding: 'utf8',
  }
);

assert.equal(result.status, 0, result.stderr);
const summary = JSON.parse(result.stdout);
assert.equal(summary.schemaVersion, 1);
assert.equal(summary.source, 'doppler');
assert.equal(summary.ok, true);
assert.equal(Array.isArray(summary.artifacts), true);
assert.equal(summary.artifacts.some((entry) => entry.id === 'kernelPath' && entry.ok === true), true);
assert.equal(summary.artifacts.some((entry) => entry.id === 'layerPattern' && entry.ok === true), true);

console.log('check-contract-artifacts.test: ok');
