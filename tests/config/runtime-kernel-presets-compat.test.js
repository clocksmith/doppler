import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const runtimePresetDir = path.join(repoRoot, 'src/config/presets/runtime/kernels');
const kernelPathDir = path.join(repoRoot, 'src/config/presets/kernel-paths');
const kernelRegistryPath = path.join(repoRoot, 'src/config/kernels/registry.json');

const kernelRegistry = JSON.parse(await fs.readFile(kernelRegistryPath, 'utf8'));

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

async function readJson(relativePath) {
  const absolutePath = path.join(repoRoot, relativePath);
  return JSON.parse(await fs.readFile(absolutePath, 'utf8'));
}

const compatibilityPresetPaths = [
  'src/config/presets/runtime/kernels/safe-q4k.json',
  'src/config/presets/runtime/kernels/dequant-f32-q4k.json',
];

for (const presetPath of compatibilityPresetPaths) {
  const preset = await readJson(presetPath);
  const policy = preset?.runtime?.inference?.kernelPathPolicy;
  const sourceScope = policy?.sourceScope ?? policy?.allowSources;
  assert.equal(policy?.mode, 'capability-aware', `${presetPath} must set kernelPathPolicy.mode=capability-aware`);
  assert.equal(policy?.onIncompatible, 'remap', `${presetPath} must set kernelPathPolicy.onIncompatible=remap`);
  assert.ok(
    Array.isArray(sourceScope) && sourceScope.includes('config'),
    `${presetPath} must allow config source for capability remap`
  );
  assert.ok(
    Array.isArray(sourceScope) && sourceScope.includes('execution-v0'),
    `${presetPath} must allow execution-v0 source for capability checks`
  );

  const kernelPathId = preset?.runtime?.inference?.kernelPath;
  assert.equal(
    kernelPathId,
    'gemma2-q4k-dequant-f32a',
    `${presetPath} must target subgroup-free gemma2-q4k-dequant-f32a`
  );

  const kernelPath = JSON.parse(
    await fs.readFile(path.join(kernelPathDir, `${kernelPathId}.json`), 'utf8')
  );
  const steps = collectKernelPathSteps(kernelPath);
  assert.ok(steps.length > 0, `${kernelPathId} must include executable steps`);

  for (const step of steps) {
    const shaderFile = String(step.kernel ?? '').trim();
    const entryPoint = String(step.entry ?? 'main').trim() || 'main';
    assert.ok(shaderFile.length > 0, `${kernelPathId}:${step.op ?? 'unknown'} missing kernel`);
    assert.ok(!shaderFile.includes('fused_'), `${kernelPathId}:${step.op} uses fused kernel ${shaderFile}`);

    const variants = getKernelVariantConfigs(shaderFile, entryPoint);
    if (variants.length === 0) {
      // softcap helpers are legacy sampling entrypoints that are not registry-indexed.
      if (step.op === 'softcap') {
        continue;
      }
      assert.fail(`${kernelPathId}:${step.op} missing registry variant for ${shaderFile}#${entryPoint}`);
    }
    const subgroupVariant = variants.find((variant) => (variant.requires ?? []).includes('subgroups'));
    assert.equal(
      subgroupVariant,
      undefined,
      `${kernelPathId}:${step.op} (${shaderFile}#${entryPoint}) must not require subgroups`
    );
  }
}

console.log('runtime-kernel-presets-compat.test: ok');
