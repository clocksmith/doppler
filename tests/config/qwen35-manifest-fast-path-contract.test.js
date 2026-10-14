import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

async function readJson(relativePath) {
  const filePath = path.join(repoRoot, relativePath);
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

// Runtime-profile execution patches are outside the conversion digest sync, so
// their added kernels must still name the current WGSL content digest.
function assertPatchDigestsMatchWgsl(profile, label) {
  const additions = profile.runtime?.inference?.executionPatch?.addKernels ?? [];
  assert.ok(additions.length > 0, `${label}: expected execution patch kernels`);
  for (const { key, kernel } of additions) {
    assert.equal(
      kernel?.digest,
      `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${kernel?.kernel}#${kernel?.entry}`]}`,
      `${label}: ${key} digest must match WGSL`
    );
  }
}

const qwen2F16PrimaryProfile = await readJson(
  'src/config/runtime/profiles/qwen-3-5-2b-f16-primary-throughput.json'
);
const qwen08MetalThroughputProfile = await readJson(
  'src/config/runtime/profiles/qwen-3-5-0-8b-metal-throughput.json'
);
const qwen08MetalParityProfile = await readJson(
  'src/config/runtime/profiles/qwen-3-5-0-8b-metal-parity.json'
);
const qwen2MetalThroughputProfile = await readJson(
  'src/config/runtime/profiles/qwen-3-5-2b-metal-throughput.json'
);

assert.equal(
  qwen2F16PrimaryProfile.runtime.inference.session,
  undefined,
  'Qwen 2B selective f16 profile must not override manifest compute-lane defaults'
);
assert.equal(
  qwen08MetalThroughputProfile.runtime.inference.compute,
  undefined,
  'Qwen 0.8B Metal throughput profile must not change compute precision'
);
assert.equal(
  qwen2MetalThroughputProfile.runtime.inference.compute,
  undefined,
  'Qwen 2B Metal throughput profile must not change compute precision'
);

assertPatchDigestsMatchWgsl(qwen08MetalThroughputProfile, 'Qwen 0.8B Metal throughput profile');
assertPatchDigestsMatchWgsl(qwen08MetalParityProfile, 'Qwen 0.8B Metal parity profile');
assertPatchDigestsMatchWgsl(qwen2MetalThroughputProfile, 'Qwen 2B Metal throughput profile');

console.log('qwen35-manifest-fast-path-contract.test: ok');
