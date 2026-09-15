import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { buildWgslClosure } from '../../src/tooling/program-bundle/wgsl-closure.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { sha256Hex } from '../../src/formats/sha256.js';

const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-source-identity-'));
const file = 'sample.wgsl', entry = 'sample_single_pass';
const source = '@compute @workgroup_size(1) fn sample_single_pass() {}\n';
const digest = `sha256:${sha256Hex(`${source}\n@@entry:${entry}`)}`;
const execution = { kernels: { sample: { kernel: file, entry, digest } }, mechanismKernels: ['sample'] };
try {
  await fs.writeFile(path.join(dir, file), source);
  const result = await buildWgslClosure(execution, [], { repoRoot: dir, kernelSourceRoot: dir });
  assert.equal(result.modules[0].digest, digest, 'supplied source bytes determine digest, independent of current registry');
  assert.equal(result.modules[0].sourceHash, `sha256:${sha256Hex(source)}`);
  execution.kernels.sample.digest = `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${file}#${entry}`]}`;
  await assert.rejects(buildWgslClosure(execution, [], { repoRoot: dir, kernelSourceRoot: dir }), /kernel digest mismatch/);
} finally { await fs.rm(dir, { recursive: true, force: true }); }
console.log('program-bundle-source-identity: passed');
