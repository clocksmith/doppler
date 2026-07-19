import assert from 'node:assert/strict';
import {
  listQuickstartModels,
  resolveQuickstartModel,
} from '../../src/client/doppler-registry.js';

const BROKEN_GEMMA_1B_QUICKSTART_REVISION = 'dfbe333a262f00050eebb6704827cad4839c6825';
const models = await listQuickstartModels();
const expectedModelIds = [
  'gemma-3-270m-it-q4k-ehf16-af32',
  'google-embeddinggemma-300m-q4k-ehf16-af32',
  'gemma-3-1b-it-q4k-ehf16-af32',
  'gemma-4-e2b-it-q4k-ehf16-af32',
  'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
  'qwen-3-5-0-8b-q4k-ehaf16',
  'qwen-3-5-2b-q4k-ehaf16',
];

// Registry should contain only hosted, release-claim-backed quickstart models.
assert.equal(models.length, expectedModelIds.length, `Expected ${expectedModelIds.length} quickstart models, got ${models.length}`);

const modelIds = models.map((m) => m.modelId);
for (const modelId of expectedModelIds) {
  assert.ok(modelIds.includes(modelId), `${modelId} missing`);
}

for (const entry of models) {
  assert.ok(entry.sourceCheckpointId, `${entry.modelId}: sourceCheckpointId missing`);
  assert.ok(entry.weightPackId, `${entry.modelId}: weightPackId missing`);
  assert.ok(entry.manifestVariantId, `${entry.modelId}: manifestVariantId missing`);
  assert.equal(entry.artifactCompleteness, 'complete');
  assert.equal(entry.runtimePromotionState, 'manifest-owned');
  assert.equal(entry.weightsRefAllowed, false);
  assert.ok(entry.classification, `${entry.modelId}: classification missing`);
  assert.ok(entry.typeCluster?.id, `${entry.modelId}: type cluster missing`);
}

// Alias resolution
{
  const entry = await resolveQuickstartModel('gemma3-1b');
  assert.equal(entry.modelId, 'gemma-3-1b-it-q4k-ehf16-af32');
  assert.ok(entry.modes.includes('text'));
  assert.ok(entry.hf);
  assert.notEqual(
    entry.hf.revision,
    BROKEN_GEMMA_1B_QUICKSTART_REVISION,
    'gemma3-1b quickstart registry must not point at the broken HF revision'
  );
}

{
  const entry = await resolveQuickstartModel('gemma4-e2b');
  assert.equal(entry.modelId, 'gemma-4-e2b-it-q4k-ehf16-af32');
  assert.ok(entry.modes.includes('text'));
  assert.ok(entry.modes.includes('vision'));
}

{
  const entry = await resolveQuickstartModel('gemma4-e2b-int4ple');
  assert.equal(entry.modelId, 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple');
  assert.ok(entry.modes.includes('text'));
  assert.ok(entry.modes.includes('vision'));
}

// HF coordinates present
{
  const entry = await resolveQuickstartModel('gemma4-e2b-int4ple');
  assert.ok(entry.hf);
  assert.equal(entry.hf.repoId, 'Clocksmith/rdrr');
  assert.ok(entry.hf.revision.length > 0);
  assert.ok(entry.hf.path.includes('gemma-4-e2b'));
}

{
  const entry = await resolveQuickstartModel('qwen3-0.8b');
  assert.equal(entry.modelId, 'qwen-3-5-0-8b-q4k-ehaf16');
  assert.ok(entry.modes.includes('text'));
  assert.equal(entry.hf.repoId, 'Clocksmith/rdrr');
  assert.ok(entry.hf.path.includes('qwen-3-5-0-8b'));
}

{
  const entry = await resolveQuickstartModel('qwen3-2b');
  assert.equal(entry.modelId, 'qwen-3-5-2b-q4k-ehaf16');
  assert.ok(entry.modes.includes('text'));
  assert.equal(entry.hf.repoId, 'Clocksmith/rdrr');
  assert.ok(entry.hf.path.includes('qwen-3-5-2b'));
}

// Unknown model throws
await assert.rejects(
  () => resolveQuickstartModel('nonexistent-model'),
  /Unknown quickstart model/
);

console.log('quickstart-registry-models.test: ok');
