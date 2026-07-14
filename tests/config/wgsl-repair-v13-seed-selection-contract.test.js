import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

function sha256File(filePath) {
  return createHash('sha256').update(readFileSync(filePath)).digest('hex');
}

const policyPath = 'tools/policies/wgsl-repair-v13-seed-selection-policy.json';
const policy = readJson(policyPath);
const state = readJson('tools/data/wgsl-repair-v13-semantic-evidence-state.json');
const mechanics = readJson('tools/data/wgsl-repair-v13-semantic-task-qualification.json');
const calibration = readJson(policy.populations.calibration.path);
const checkpointSelection = readJson(policy.populations.checkpointSelection.path);

assert.equal(policy.status, 'frozen_before_candidate_evaluation');
assert.equal(policy.eligibleLane, 'external20');
assert.deepEqual(policy.eligibleCandidates.map((entry) => entry.seed), [11, 29, 47]);
assert.equal(policy.promptContract.contextCharactersPerSide, 600);
assert.equal(policy.promptContract.useChatTemplate, false);
assert.equal(policy.generation.temperature, 0);
assert.equal(policy.generation.topK, 1);
assert.deepEqual(policy.ranking.lexicographicOrder, [
  'semantic_task_pass_count_desc',
  'compile_pass_count_desc',
  'semantic_variant_pass_count_desc',
  'exact_reference_completion_count_desc',
  'seed_asc',
]);

assert.equal(sha256File(policy.populations.calibration.path), policy.populations.calibration.sha256);
assert.equal(
  sha256File(policy.populations.checkpointSelection.path),
  policy.populations.checkpointSelection.sha256
);
assert.equal(
  sha256File(policy.predecessor.adapterPortabilityReceiptPath),
  policy.predecessor.adapterPortabilityReceiptSha256
);
assert.equal(
  sha256File(policy.runtime.artifactStatusPath),
  policy.runtime.artifactStatusSha256
);
assert.equal(
  sha256File(policy.runtime.conversionConfigPath),
  policy.runtime.conversionConfigSha256
);
for (const candidate of policy.eligibleCandidates) {
  assert.equal(sha256File(candidate.adapterManifestPath), candidate.adapterManifestSha256);
  assert.equal(sha256File(candidate.adapterWeightsPath), candidate.adapterWeightsSha256);
}

assert.equal(calibration.role, 'calibration');
assert.equal(calibration.populationAuthority, 'prompt_and_harness_calibration_only');
assert.equal(checkpointSelection.role, 'checkpoint_selection');
assert.equal(checkpointSelection.populationAuthority, 'external20_seed_selection_only');
assert.equal(state.populations.calibration.status, 'frozen');
assert.equal(state.populations.calibration.populationHash, policy.populations.calibration.sha256);
assert.equal(state.populations.checkpointSelection.status, 'frozen');
assert.equal(
  state.populations.checkpointSelection.populationHash,
  policy.populations.checkpointSelection.sha256
);
assert.equal(state.populations.seedConfirmation.status, 'unmaterialized');
assert.equal(state.populations.promotion.status, 'unmaterialized');

const manifests = [mechanics, calibration, checkpointSelection];
const familyRoles = new Map();
const sourceRoles = new Map();
const inputSeedRoles = new Map();
for (const manifest of manifests) {
  for (const task of manifest.tasks) {
    assert.ok(!familyRoles.has(task.kernelFamilyId), `${task.kernelFamilyId} overlaps roles`);
    assert.ok(!sourceRoles.has(task.sourcePath), `${task.sourcePath} overlaps roles`);
    assert.ok(!inputSeedRoles.has(task.inputSeed), `${task.inputSeed} overlaps roles`);
    familyRoles.set(task.kernelFamilyId, manifest.role);
    sourceRoles.set(task.sourcePath, manifest.role);
    inputSeedRoles.set(task.inputSeed, manifest.role);
    assert.equal(sha256File(task.sourcePath), task.sourceSha256);
    const source = readFileSync(task.sourcePath, 'utf8').trim();
    assert.equal(source.split(task.brokenSpan).length - 1, 1);
    assert.equal(source.includes(task.referenceSpan), false);
    assert.equal(task.variants.length, 3);
    assert.equal(new Set(task.variants.map((variant) => variant.workgroupId)).size, 2);
  }
}

console.log('wgsl-repair-v13-seed-selection-contract.test: ok');
