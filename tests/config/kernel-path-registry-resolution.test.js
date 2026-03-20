import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { getKernelPath, resolveKernelPath } from '../../src/config/kernel-path-loader.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const registryPath = path.join(repoRoot, 'src/config/kernel-paths/registry.json');
const kernelRegistryPath = path.join(repoRoot, 'src/config/kernels/registry.json');

const kernelPathRegistry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
const kernelRegistry = JSON.parse(await fs.readFile(kernelRegistryPath, 'utf8'));
const entries = Array.isArray(kernelPathRegistry?.entries) ? kernelPathRegistry.entries : [];

function getKernelVariantConfigs(shaderFile, entryPoint) {
  const matches = [];
  for (const operation of Object.values(kernelRegistry.operations ?? {})) {
    for (const variant of Object.values(operation.variants ?? {})) {
      if (variant?.wgsl !== shaderFile) continue;
      const variantEntry = variant?.entryPoint ?? 'main';
      if (variantEntry !== entryPoint) continue;
      matches.push(variant);
    }
  }
  return matches;
}

function collectKernelPathSteps(kernelPath) {
  const steps = [];
  const pushSteps = (list) => {
    for (const step of list ?? []) {
      if (step && typeof step === 'object') {
        steps.push(step);
      }
    }
  };
  pushSteps(kernelPath.decode?.steps);
  pushSteps(kernelPath.prefill?.steps);
  pushSteps(kernelPath.preLayer);
  pushSteps(kernelPath.postLayer);
  pushSteps(kernelPath.sampling);
  for (const override of kernelPath.layerOverrides ?? []) {
    pushSteps(override.steps);
  }
  return steps;
}

for (const entry of entries) {
  const resolved = resolveKernelPath(entry.id);
  assert.ok(resolved, `registry entry "${entry.id}" must resolve`);

  if (typeof entry.aliasOf === 'string' && entry.aliasOf.length > 0) {
    const aliasTarget = resolveKernelPath(entry.aliasOf);
    assert.equal(
      resolved,
      aliasTarget,
      `alias entry "${entry.id}" must resolve to the same kernel path object as "${entry.aliasOf}"`
    );
    continue;
  }

  const direct = getKernelPath(entry.id);
  assert.equal(direct, resolved, `registry entry "${entry.id}" must be addressable via getKernelPath()`);
  assert.equal(resolved.id, entry.id, `file-backed registry entry "${entry.id}" must preserve its own id`);

  const steps = collectKernelPathSteps(resolved);
  assert.ok(steps.length > 0, `${entry.id} must include executable steps`);

  for (const step of steps) {
    const shaderFile = String(step.kernel ?? '').trim();
    const entryPoint = String(step.entry ?? 'main').trim() || 'main';
    assert.ok(shaderFile.length > 0, `${entry.id}:${step.op ?? 'unknown'} missing kernel`);

    const variants = getKernelVariantConfigs(shaderFile, entryPoint);
    if (variants.length === 0) {
      if (step.op === 'softcap') {
        continue;
      }
      assert.fail(`${entry.id}:${step.op} missing registry variant for ${shaderFile}#${entryPoint}`);
    }
  }
}

console.log('kernel-path-registry-resolution.test: ok');
