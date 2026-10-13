// Explicit isolated consumer only; imports the installed public package and opens no device.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
const consumer = process.argv[2]; assert(consumer, 'Explicit installed consumer directory required');
const require = createRequire(resolve(consumer, 'package.json'));
const runtime = await import(pathToFileURL(require.resolve('doppler-gpu/partitions')));
for (const count of [2, 3, 4, 7]) {
  const plan = runtime.createOrderedLayerPartitionPlan({ modelId: 'installed', numLayers: 28, hiddenSize: 4, vocabSize: 16,
    splitLayers: Array.from({ length: count - 1 }, (_, index) => Math.floor(28 * (index + 1) / count)), activationPrecision: 'two-f32' });
  assert.equal(plan.stageCount, count); assert.match(runtime.hashLayerPartitionPlan(plan), /^sha256:/);
  const values = new Float32Array([1, 1e-8, 2, 1e-8, 3, 1e-8, 4, 1e-8]);
  const frame = runtime.serializeActivationFrame({ shape: [1, 1, 4], dtype: 'f32', precision: 'two-f32', components: 2, elementStrideBytes: 8, data: values });
  assert.deepEqual(runtime.validateActivationFrameForPlan(frame, plan).tensorData, values);
  assert.throws(() => runtime.validateActivationFrameForPlan({ ...frame, elementStrideBytes: 4 }, plan));
}
assert.equal(typeof runtime.createManifestResidentPartitionFactory, 'function');
assert.match(runtime.hashPartitionGeneration({ maxTokens: 2, maxSeqLen: 8, temperature: 0, topK: 1, topP: 1,
  repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false }), /^sha256:/);
console.log('Installed public ordered plans, exact activation bits, factory and generation contracts passed; no GPU opened');
