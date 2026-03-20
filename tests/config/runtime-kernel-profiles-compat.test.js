import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const runtimeProfileDir = path.join(repoRoot, 'src/config/runtime/kernels');
const kernelPathDir = path.join(repoRoot, 'src/config/kernel-paths');
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

const canonicalProfilePaths = [
  'src/config/runtime/kernels/gemma2-q4k-dequant-f16a.json',
  'src/config/runtime/kernels/gemma2-q4k-dequant-f32a-nosubgroups.json',
  'src/config/runtime/kernels/gemma2-q4k-fused-f32a.json',
  'src/config/runtime/kernels/embeddinggemma-q4k-dequant-f32a.json',
];

const deprecatedAliasProfilePaths = [
  'src/config/runtime/kernels/safe-q4k.json',
  'src/config/runtime/kernels/dequant-f32-q4k.json',
  'src/config/runtime/kernels/fused-q4k.json',
  'src/config/runtime/kernels/dequant-f16-q4k.json',
];

const subgroupFreeProfilePaths = [
  'src/config/runtime/kernels/gemma2-q4k-dequant-f32a-nosubgroups.json',
];

for (const profilePath of canonicalProfilePaths) {
  const profile = await readJson(profilePath);
  const kernelPathId = profile?.runtime?.inference?.kernelPath;
  const expectedProfileId = `kernels/${kernelPathId}`;
  const expectedName = kernelPathId;
  assert.equal(profile?.stability, 'canonical', `${profilePath} must be canonical`);
  assert.equal(profile?.id, expectedProfileId, `${profilePath} must use id ${expectedProfileId}`);
  assert.equal(profile?.name, expectedName, `${profilePath} must use name ${expectedName}`);
  assert.ok(
    !String(profile?.description ?? '').toLowerCase().includes('safe'),
    `${profilePath} must not advertise semantic "safe" behavior`
  );
}

for (const profilePath of deprecatedAliasProfilePaths) {
  const profile = await readJson(profilePath);
  assert.equal(profile?.stability, 'deprecated', `${profilePath} must be deprecated`);
  assert.ok(typeof profile?.replacementId === 'string' && profile.replacementId.length > 0, `${profilePath} must declare replacementId`);
  assert.equal(profile?.extends, profile?.replacementId, `${profilePath} must extend its replacement profile`);
  assert.deepEqual(profile?.runtime ?? null, {}, `${profilePath} must delegate runtime config to its replacement`);
}

for (const profilePath of subgroupFreeProfilePaths) {
  const profile = await readJson(profilePath);
  const policy = profile?.runtime?.inference?.kernelPathPolicy;
  const sourceScope = policy?.sourceScope ?? policy?.allowSources;
  assert.equal(policy?.mode, 'capability-aware', `${profilePath} must set kernelPathPolicy.mode=capability-aware`);
  assert.equal(policy?.onIncompatible, 'remap', `${profilePath} must set kernelPathPolicy.onIncompatible=remap`);
  assert.ok(
    Array.isArray(sourceScope) && sourceScope.includes('config'),
    `${profilePath} must allow config source for capability remap`
  );
  assert.ok(
    Array.isArray(sourceScope) && sourceScope.includes('model'),
    `${profilePath} must allow model source for capability checks`
  );

  const kernelPathId = profile?.runtime?.inference?.kernelPath;
  assert.equal(
    kernelPathId,
    'gemma2-q4k-dequant-f32a-nosubgroups',
    `${profilePath} must target subgroup-free gemma2-q4k-dequant-f32a-nosubgroups`
  );
}

for (const profilePath of canonicalProfilePaths) {
  const profile = await readJson(profilePath);
  const kernelPathId = profile?.runtime?.inference?.kernelPath;

  const kernelPath = JSON.parse(
    await fs.readFile(path.join(kernelPathDir, `${kernelPathId}.json`), 'utf8')
  );
  const expectedActivationDtype = kernelPath.activationDtype;
  const expectedKvDtype = kernelPath.kvDtype ?? expectedActivationDtype;
  const expectedOutputDtype = kernelPath.outputDtype ?? expectedActivationDtype;
  assert.equal(
    profile?.runtime?.inference?.compute?.activationDtype,
    expectedActivationDtype,
    `${profilePath} must set runtime.inference.compute.activationDtype=${expectedActivationDtype}`
  );
  assert.equal(
    profile?.runtime?.inference?.kvcache?.kvDtype,
    expectedKvDtype,
    `${profilePath} must set runtime.inference.kvcache.kvDtype=${expectedKvDtype}`
  );
  assert.equal(
    profile?.runtime?.inference?.session?.compute?.defaults?.outputDtype,
    expectedOutputDtype,
    `${profilePath} must set runtime.inference.session.compute.defaults.outputDtype=${expectedOutputDtype}`
  );
}

for (const profilePath of subgroupFreeProfilePaths) {
  const profile = await readJson(profilePath);
  const kernelPathId = profile?.runtime?.inference?.kernelPath;
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

console.log('runtime-kernel-profiles-compat.test: ok');
