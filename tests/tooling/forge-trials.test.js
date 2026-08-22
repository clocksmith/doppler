import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { stageAnalyze, stageInspect, stagePackage, stageSpecialize } from '../../src/converter/forge-stages.js';
import { validatePackV2 } from '../../src/tooling/pack-v2.js';

// Test 1: Qwen3.8-27B Lineage Acceleration Trial
const qwenConfigRaw = await fs.readFile(
  path.resolve('src/config/conversion/qwen38/qwen3.8-27b-forge.json'),
  'utf8'
);
const qwenConfig = JSON.parse(qwenConfigRaw);
assert.equal(qwenConfig.modelId, 'qwen-3.8-27b');

const qwenIntake = await stageInspect({
  modelDir: '/models/qwen-3.8-27b',
  config: qwenConfig.source,
});
const qwenAnalyzed = stageAnalyze(qwenIntake.data);
assert.equal(qwenAnalyzed.modelIR.hiddenSize, 5120);
assert.equal(qwenAnalyzed.modelIR.numLayers, 64);

const qwenSpecialized = stageSpecialize(qwenAnalyzed.modelIR, [
  { id: 'qwen_gemm', file: 'matmul.wgsl', entry: 'main', digest: `sha256:${'0'.repeat(64)}` },
]);
assert.equal(qwenSpecialized.targetPlans.length, 3);

const qwenPackResult = stagePackage({
  modelIR: qwenAnalyzed.modelIR,
  targetPlans: qwenSpecialized.targetPlans,
  wgslModules: [{ id: 'qwen_gemm', file: 'matmul.wgsl', entry: 'main', digest: `sha256:${'0'.repeat(64)}` }],
  artifacts: [{ role: 'manifest', path: 'manifest.json', hash: `sha256:${'a'.repeat(64)}`, sizeBytes: 100 }],
});
assert.equal(qwenPackResult.ok, true);
assert.equal(validatePackV2(qwenPackResult.pack).ok, true);

// Test 2: Meta Muse Glimmer 30B Generalization Trial
const museConfigRaw = await fs.readFile(
  path.resolve('src/config/conversion/muse/muse-glimmer-30b-forge.json'),
  'utf8'
);
const museConfig = JSON.parse(museConfigRaw);
assert.equal(museConfig.modelId, 'muse-glimmer-30b');

const museIntake = await stageInspect({
  modelDir: '/models/muse-glimmer-30b',
  config: museConfig.source,
});
const museAnalyzed = stageAnalyze(museIntake.data);
assert.equal(museAnalyzed.modelIR.hiddenSize, 6144);
assert.equal(museAnalyzed.modelIR.numLayers, 48);

const museSpecialized = stageSpecialize(museAnalyzed.modelIR, [
  { id: 'muse_norm', file: 'rmsnorm.wgsl', entry: 'main', digest: `sha256:${'1'.repeat(64)}` },
]);
assert.equal(museSpecialized.targetPlans.length, 3);

const musePackResult = stagePackage({
  modelIR: museAnalyzed.modelIR,
  targetPlans: museSpecialized.targetPlans,
  wgslModules: [{ id: 'muse_norm', file: 'rmsnorm.wgsl', entry: 'main', digest: `sha256:${'1'.repeat(64)}` }],
  artifacts: [{ role: 'manifest', path: 'manifest.json', hash: `sha256:${'b'.repeat(64)}`, sizeBytes: 100 }],
});
assert.equal(musePackResult.ok, true);
assert.equal(validatePackV2(musePackResult.pack).ok, true);

console.log('✔ forge-trials.test.js passed');
