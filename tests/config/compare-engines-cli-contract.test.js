import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { mergeKernelPathPolicy } from '../../src/config/merge-helpers.js';
import {
  buildSharedBenchmarkContract,
  parseArgs as parseCompareArgs,
  parseOnOff as parseCompareOnOff,
} from '../../tools/compare-engines.js';

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
  const quickstartRegistryPath = path.join(repoRoot, 'src', 'client', 'doppler-registry.json');
  const compareConfig = JSON.parse(await fs.readFile(compareConfigPath, 'utf8'));
  const benchmarkPolicy = JSON.parse(await fs.readFile(benchmarkPolicyPath, 'utf8'));
  const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
  const quickstartRegistry = JSON.parse(await fs.readFile(quickstartRegistryPath, 'utf8'));
  const catalogIds = new Set((Array.isArray(catalog.models) ? catalog.models : []).map((entry) => entry?.modelId).filter(Boolean));
  const quickstartIds = new Set((Array.isArray(quickstartRegistry.models) ? quickstartRegistry.models : []).map((entry) => entry?.modelId).filter(Boolean));
  const compareProfileIds = new Set(
    (Array.isArray(compareConfig.modelProfiles) ? compareConfig.modelProfiles : [])
      .map((entry) => entry?.dopplerModelId)
      .filter(Boolean)
  );
  const qwen08Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'qwen-3-5-0-8b-q4k-ehaf16') || null;
  const qwen2Profile = compareConfig.modelProfiles.find((entry) => entry?.dopplerModelId === 'qwen-3-5-2b-q4k-ehaf16') || null;

  assert.ok(qwen08Profile, 'compare config must include qwen-3-5-0-8b-q4k-ehaf16');
  assert.equal(qwen08Profile.defaultDopplerSurface, 'browser');
  assert.equal(qwen08Profile.compareLane, 'performance_comparable');
  assert.equal(qwen08Profile.compareLaneReason, null);

  assert.ok(qwen2Profile, 'compare config must include qwen-3-5-2b-q4k-ehaf16');
  assert.equal(qwen2Profile.compareLane, 'capability_only');
  assert.match(qwen2Profile.compareLaneReason, /not yet promoted to a claimable compare lane/i);

  for (const profile of compareConfig.modelProfiles) {
    assert.ok(['performance_comparable', 'capability_only'].includes(profile.compareLane));
    if (profile.compareLane === 'capability_only') {
      assert.equal(typeof profile.compareLaneReason, 'string');
      assert.ok(profile.compareLaneReason.length > 0);
    }

    if (profile.defaultDopplerSource === 'quickstart-registry') {
      assert.equal(profile.modelBaseDir, null);
      assert.ok(
        quickstartIds.has(profile.dopplerModelId),
        `${profile.dopplerModelId}: quickstart compare profiles must exist in src/client/doppler-registry.json`
      );
      continue;
    }

    assert.equal(profile.defaultDopplerSource, 'local');
    if (profile?.modelBaseDir !== 'local') {
      continue;
    }
    const manifestPath = path.join(repoRoot, 'models', 'local', profile.dopplerModelId, 'manifest.json');
    let manifest = null;
    try {
      manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    } catch (error) {
      if (error?.code !== 'ENOENT') {
        throw error;
      }
    }
    if (manifest) {
      assert.equal(manifest.modelId, profile.dopplerModelId);
      if (profile.defaultKernelPath != null) {
        assert.equal(
          manifest?.inference?.defaultKernelPath ?? null,
          profile.defaultKernelPath,
          `${profile.dopplerModelId}: compare profile defaultKernelPath must match the local manifest defaultKernelPath`
        );
      }
      continue;
    }
    assert.equal(typeof profile.dopplerModelId, 'string');
    assert.ok(profile.dopplerModelId.trim().length > 0);
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
      localExists || catalogIds.has(modelId) || compareProfileIds.has(modelId),
      `benchmark-policy knownBadByModel.${modelId} must resolve to a local manifest, compare profile, or catalog model`
    );
  }

  const compareRuntimePolicy = mergeKernelPathPolicy(
    undefined,
    benchmarkPolicy?.kernelPathPolicy?.compareRuntime
  );
  assert.equal(compareRuntimePolicy.mode, 'capability-aware');
  assert.deepEqual(compareRuntimePolicy.sourceScope, ['model', 'manifest', 'config']);
  assert.deepEqual(compareRuntimePolicy.allowSources, ['model', 'manifest', 'config']);
  assert.equal(compareRuntimePolicy.onIncompatible, 'remap');
  assert.equal(benchmarkPolicy?.browser?.compareChannelByPlatform?.darwin, 'chromium');
  assert.equal(benchmarkPolicy?.browser?.compareChannelByPlatform?.linux, 'chromium');
  assert.equal(benchmarkPolicy?.browser?.compareChannelByPlatform?.win32, 'chromium');
}

{
  const result = runCompareEngines(['--help']);
  assert.equal(result.status, 0, result.stderr);
}

{
  const flags = parseCompareArgs(['--use-chat-template', 'off']);
  assert.equal(flags['use-chat-template'], 'off');
  const sharedContract = buildSharedBenchmarkContract({
    prompt: 'unit prompt',
    maxTokens: 4,
    warmupRuns: 0,
    timedRuns: 1,
    seed: 7,
    sampling: {
      temperature: 0,
      topK: 1,
      topP: 1,
    },
    useChatTemplate: parseCompareOnOff(flags['use-chat-template'], false, '--use-chat-template'),
  });
  assert.equal(sharedContract.useChatTemplate, false);
  assert.equal(sharedContract.sampling.repetitionPenalty, 1);
  assert.equal(sharedContract.sampling.greedyThreshold, 0.01);
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
        defaultDopplerSource: 'local',
        modelBaseDir: 'local',
        defaultDopplerSurface: 'unsupported',
        compareLane: 'performance_comparable',
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

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const nonComparableConfigPath = path.join(tempDir, 'capability-only-compare-config.json');
  const nonComparableConfig = {
    schemaVersion: 1,
    updated: '2026-03-27',
    modelProfiles: [
      {
        dopplerModelId: 'unit-capability-model',
        defaultTjsModelId: null,
        defaultKernelPath: null,
        defaultDopplerSource: 'local',
        modelBaseDir: 'local',
        defaultDopplerSurface: 'auto',
        defaultDopplerFormat: 'rdrr',
        defaultTjsFormat: null,
        safetensorsSourceId: null,
        compareLane: 'capability_only',
        compareLaneReason: 'unit-test support-only lane',
      },
    ],
  };
  await fs.writeFile(nonComparableConfigPath, `${JSON.stringify(nonComparableConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', nonComparableConfigPath,
    '--model-id', 'unit-capability-model',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /allow-non-comparable-lane/);
}

console.log('compare-engines-cli-contract.test: ok');
