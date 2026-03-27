import assert from 'node:assert/strict';
import {
  listQuickstartModels,
  resolveQuickstartModel,
} from '../../src/client/doppler-registry.js';

const models = await listQuickstartModels();

// Registry should have 5 models after adding Gemma 3 1B, Qwen 3.5 0.8B, Qwen 3.5 2B
assert.equal(models.length, 5, `Expected 5 quickstart models, got ${models.length}`);

const modelIds = models.map((m) => m.modelId);
assert.ok(modelIds.includes('gemma-3-270m-it-q4k-ehf16-af32'), 'Gemma 3 270M missing');
assert.ok(modelIds.includes('google-embeddinggemma-300m-q4k-ehf16-af32'), 'EmbeddingGemma 300M missing');
assert.ok(modelIds.includes('gemma-3-1b-it-q4k-ehf16-af32'), 'Gemma 3 1B missing');
assert.ok(modelIds.includes('qwen-3-5-0-8b-q4k-ehaf16'), 'Qwen 3.5 0.8B missing');
assert.ok(modelIds.includes('qwen-3-5-2b-q4k-ehaf16'), 'Qwen 3.5 2B missing');

// Alias resolution
{
  const entry = await resolveQuickstartModel('gemma3-1b');
  assert.equal(entry.modelId, 'gemma-3-1b-it-q4k-ehf16-af32');
  assert.ok(entry.modes.includes('text'));
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
