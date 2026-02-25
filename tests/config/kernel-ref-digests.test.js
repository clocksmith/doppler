import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { sha256Hex } from '../../src/utils/sha256.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const registryPath = path.join(repoRoot, 'src/config/kernels/registry.json');
const kernelsDir = path.join(repoRoot, 'src/gpu/kernels');

const registry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
const expectedDigests = new Map();

for (const opSchema of Object.values(registry.operations ?? {})) {
  for (const variant of Object.values(opSchema.variants ?? {})) {
    const wgsl = variant?.wgsl;
    const entry = variant?.entryPoint ?? 'main';
    if (typeof wgsl !== 'string' || wgsl.length === 0) {
      continue;
    }
    const key = `${wgsl}#${entry}`;
    if (expectedDigests.has(key)) {
      continue;
    }
    const shaderPath = path.join(kernelsDir, wgsl);
    const shaderSource = (await fs.readFile(shaderPath, 'utf8')).replace(/\r\n/g, '\n');
    expectedDigests.set(key, sha256Hex(`${shaderSource}\n@@entry:${entry}`));
  }
}

assert.equal(
  Object.keys(KERNEL_REF_CONTENT_DIGESTS).length,
  expectedDigests.size,
  'kernel-ref digest registry size mismatch'
);

for (const [key, expected] of expectedDigests.entries()) {
  assert.equal(
    KERNEL_REF_CONTENT_DIGESTS[key],
    expected,
    `kernel-ref content digest mismatch for ${key}`
  );
}

for (const key of Object.keys(KERNEL_REF_CONTENT_DIGESTS)) {
  assert.ok(expectedDigests.has(key), `kernel-ref digest contains unknown key ${key}`);
}

