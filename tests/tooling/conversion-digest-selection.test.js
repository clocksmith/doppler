import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-candidate-digests-'));
const run = (...args) => spawnSync(process.execPath, ['tools/sync-conversion-kernel-digests.js', ...args], { encoding: 'utf8' });
try {
  const candidate = path.join(root, 'candidate.json');
  const previous = path.join(root, 'previous.json');
  const value = { execution: { kernels: { projection: {
    kernel: 'fused_matmul_q4.wgsl', entry: 'main_gemv', digest: `sha256:${'0'.repeat(64)}`,
  } } } };
  const original = JSON.stringify(value);
  await fs.writeFile(candidate, original);
  await fs.writeFile(previous, original);
  assert.notEqual(run('--file', candidate, '--check').status, 0);
  assert.equal(await fs.readFile(candidate, 'utf8'), original, 'checking must not mutate recipes');
  const sync = run('--file', candidate);
  assert.equal(sync.status, 0, sync.stderr);
  assert.equal(await fs.readFile(previous, 'utf8'), original, 'selected sync must not rewrite adjacent historical candidates');
  assert.equal(JSON.parse(await fs.readFile(candidate, 'utf8')).execution.kernels.projection.digest,
    `sha256:${KERNEL_REF_CONTENT_DIGESTS['fused_matmul_q4.wgsl#main_gemv']}`);
  assert.equal(run('--file', candidate, '--check').status, 0);
  assert.notEqual(run('--file').status, 0);
  assert.notEqual(run('--unknown').status, 0);
  await fs.writeFile(candidate, '{');
  assert.notEqual(run('--file', candidate, '--check').status, 0, 'selected malformed inputs must not report success');
} finally {
  await fs.rm(root, { recursive: true, force: true });
}
console.log('conversion-digest-selection.test: ok');
