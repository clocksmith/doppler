import assert from 'node:assert/strict';
import {
  listQuickstartModels,
  resolveQuickstartModel,
} from '../../src/client/doppler-registry.js';

const BROKEN_GEMMA_1B_QUICKSTART_REVISION = 'dfbe333a262f00050eebb6704827cad4839c6825';
const models = await listQuickstartModels();

// Registry should have 7 quickstart models, including the hosted Gemma 4 INT4 PLE variant.
assert.equal(models.length, 7, `Expected 7 quickstart models, got ${models.length}`);

const modelIds = models.map((m) => m.modelId);
assert.ok(modelIds.includes('gemma-3-270m-it-q4k-ehf16-af32'), 'Gemma 3 270M missing');
assert.ok(modelIds.includes('google-embeddinggemma-300m-q4k-ehf16-af32'), 'EmbeddingGemma 300M missing');
assert.ok(modelIds.includes('gemma-3-1b-it-q4k-ehf16-af32'), 'Gemma 3 1B missing');
assert.ok(modelIds.includes('gemma-4-e2b-it-q4k-ehf16-af32'), 'Gemma 4 E2B missing');
assert.ok(modelIds.includes('gemma-4-e2b-it-q4k-ehf16-af32-int4ple'), 'Gemma 4 E2B INT4 PLE missing');
assert.ok(modelIds.includes('qwen-3-5-0-8b-q4k-ehaf16'), 'Qwen 3.5 0.8B missing');
assert.ok(modelIds.includes('qwen-3-5-2b-q4k-ehaf16'), 'Qwen 3.5 2B missing');

for (const entry of models) {
  assert.ok(entry.sourceCheckpointId, `${entry.modelId}: sourceCheckpointId missing`);
  assert.ok(entry.weightPackId, `${entry.modelId}: weightPackId missing`);
  assert.ok(entry.manifestVariantId, `${entry.modelId}: manifestVariantId missing`);
  assert.equal(entry.artifactCompleteness, 'complete');
  assert.equal(entry.runtimePromotionState, 'manifest-owned');
  assert.equal(entry.weightsRefAllowed, false);
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

{
  const entry = await resolveQuickstartModel('qwen3-0.8b');
  assert.equal(entry.modelId, 'qwen-3-5-0-8b-q4k-ehaf16');
  assert.ok(entry.modes.includes('text'));
}

{
  const entry = await resolveQuickstartModel('qwen3-2b');
  assert.equal(entry.modelId, 'qwen-3-5-2b-q4k-ehaf16');
  assert.ok(entry.modes.includes('text'));
}

// HF coordinates present
{
  const entry = await resolveQuickstartModel('qwen3-0.8b');
  assert.ok(entry.hf);
  assert.equal(entry.hf.repoId, 'Clocksmith/rdrr');
  assert.ok(entry.hf.revision.length > 0);
  assert.ok(entry.hf.path.includes('qwen'));
}

// Unknown model throws
await assert.rejects(
  () => resolveQuickstartModel('nonexistent-model'),
  /Unknown quickstart model/
);

console.log('quickstart-registry-models.test: ok');
