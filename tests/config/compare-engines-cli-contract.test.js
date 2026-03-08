import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';

function runCompareEngines(args) {
  return spawnSync(process.execPath, ['tools/compare-engines.js', ...args], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
}

{
  const repoRoot = process.cwd();
  const compareConfigPath = path.join(repoRoot, 'benchmarks', 'vendors', 'compare-engines.config.json');
  const benchmarkPolicyPath = path.join(repoRoot, 'benchmarks', 'vendors', 'benchmark-policy.json');
  const catalogPath = path.join(repoRoot, 'models', 'catalog.json');
  const compareConfig = JSON.parse(await fs.readFile(compareConfigPath, 'utf8'));
  const benchmarkPolicy = JSON.parse(await fs.readFile(benchmarkPolicyPath, 'utf8'));
  const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
  const catalogIds = new Set((Array.isArray(catalog.models) ? catalog.models : []).map((entry) => entry?.modelId).filter(Boolean));

  for (const profile of compareConfig.modelProfiles) {
    if (profile?.modelBaseDir !== 'local') {
      continue;
    }
    const manifestPath = path.join(repoRoot, 'models', 'local', profile.dopplerModelId, 'manifest.json');
    await assert.doesNotReject(
      fs.access(manifestPath),
      `${profile.dopplerModelId}: local compare profile must resolve to models/local/<modelId>/manifest.json`
    );
    const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    assert.equal(manifest.modelId, profile.dopplerModelId);
    if (profile.defaultKernelPath != null) {
      assert.equal(
        manifest?.inference?.defaultKernelPath ?? null,
        profile.defaultKernelPath,
        `${profile.dopplerModelId}: compare profile defaultKernelPath must match the local manifest defaultKernelPath`
      );
    }
  }

  const knownBadByModel = benchmarkPolicy?.kernelPathPolicy?.knownBadByModel ?? {};
  for (const modelId of Object.keys(knownBadByModel)) {
    const localManifestPath = path.join(repoRoot, 'models', 'local', modelId, 'manifest.json');
    let localExists = true;
    try {
      await fs.access(localManifestPath);
    } catch {
      localExists = false;
    }
    assert.ok(
      localExists || catalogIds.has(modelId),
      `benchmark-policy knownBadByModel.${modelId} must resolve to a local manifest or a catalog model`
    );
  }
}

{
  const result = runCompareEngines(['--help']);
  assert.equal(result.status, 0, result.stderr);
}

{
  const result = runCompareEngines([
    '--doppler-surface', 'invalid-surface',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
}

{
  const result = runCompareEngines([
    '--runtime-config-json',
    '{"inference":{"prompt":"override"}}',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(
    result.stderr,
    /--runtime-config-json must not override compare-managed fairness or cadence fields/
  );
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const badConfigPath = path.join(tempDir, 'bad-compare-config.json');
  const badConfig = {
    schemaVersion: 1,
    updated: '2026-03-05',
    modelProfiles: [
      {
        dopplerModelId: 'gemma-3-270m-it-f16-af32',
        defaultTjsModelId: 'onnx-community/gemma-3-270m-it-ONNX',
        defaultKernelPath: null,
        modelBaseDir: 'local',
        defaultDopplerSurface: 'unsupported',
      },
    ],
  };
  await fs.writeFile(badConfigPath, `${JSON.stringify(badConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', badConfigPath,
    '--json',
  ]);
  assert.notEqual(result.status, 0);
}

console.log('compare-engines-cli-contract.test: ok');
