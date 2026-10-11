import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { resolvePhysicalPartitionConfig, bindPhysicalIndependentReference, physicalInputDigest,
  INDEPENDENT_TOLERANCE } from '../helpers/resident-partition-physical-config.js';

const manifest = { modelId: 'physical-fixture', architecture: { numLayers: 28, hiddenSize: 8, vocabSize: 4 },
  inference: { session: { compute: { defaults: { activationDtype: 'f32' } } } } };
const identity = physicalInputDigest(JSON.stringify(manifest));
const original = resolvePhysicalPartitionConfig(manifest, identity);
assert.equal(original.plan.splitLayer, 14);
assert.equal(original.generation.maxTokens, 3);
assert.equal(original.generation.presencePenalty, 0.1);
assert.equal(original.limits.maxOutputCharacters, 1024);
assert.deepEqual(original.prompts, ['The color of the sky is', 'The capital of France is']);
const chat = resolvePhysicalPartitionConfig(manifest, identity, null, { chat: true });
assert.equal(chat.generation.maxTokens, 1024);
assert.equal(chat.generation.useChatTemplate, true);
const config = { schema: 'doppler.resident-physical-config/v1', modelIdentity: identity, splitLayer: 11,
  generation: { ...original.generation, temperature: 0.6, topK: 2, seed: 17 },
  limits: { ...original.limits, maxOutputCharacters: 16000 }, prompts: ['Pinned prompt'], reference: null };
const resolved = resolvePhysicalPartitionConfig(manifest, identity, config);
assert.deepEqual(resolved.plan.partitions.map(partition => partition.layerRange), [[0, 10], [11, 27]]);
assert.equal(resolved.limits.maxOutputCharacters, 16000);
assert.equal(resolved.generation.seed, 17);
assert.equal(INDEPENDENT_TOLERANCE, 0.001);
for (const mutate of [
  value => { value.modelIdentity = 'sha256:' + '0'.repeat(64); },
  value => { value.schema = 'other'; },
  value => { value.splitLayer = 28; },
  value => { value.splitLayer = null; },
  value => { value.unrecognized = true; },
  value => { value.generation.unrecognized = true; },
  value => { value.generation = null; },
  value => { value.generation = undefined; },
  value => { delete value.generation.seed; },
  value => { value.limits.maxTokens = 1; },
  value => { value.limits.maxConcurrentAttempts = 17; },
  value => { value.limits.maxAttempts = 3; },
  value => { value.limits.maxPromptTokens = 0; },
  value => { value.prompts = []; },
  value => { value.prompts = null; },
  value => { value.prompts = undefined; },
  value => { value.limits = null; },
  value => { value.limits = undefined; },
  value => { value.splitLayer = undefined; },
  value => { value.reference = undefined; },
  value => { value.prompts = [[{ role: 'invalid', content: 'text' }]]; },
  value => { value.reference = { path: 'reference.json', sha256: 'unpinned' }; },
]) {
  const invalid = structuredClone(config); mutate(invalid);
  assert.throws(() => resolvePhysicalPartitionConfig(manifest, identity, invalid));
}
assert.throws(() => resolvePhysicalPartitionConfig(manifest, identity, config, { chat: true }), /cannot mix/);
assert.throws(() => resolvePhysicalPartitionConfig(manifest, identity, config, { prompts: ['other'] }), /cannot mix/);
const directory = mkdtempSync(join(tmpdir(), 'doppler-physical-config-'));
try {
  writeFileSync(join(directory, 'manifest.json'), JSON.stringify(manifest));
  writeFileSync(join(directory, 'config.json'), 'null');
  const child = spawnSync(process.execPath, [new URL('./resident-partition-physical.test.js', import.meta.url).pathname], {
    encoding: 'utf8', env: { ...process.env, DOPPLER_PARTITION_MODEL_DIR: directory,
      DOPPLER_PARTITION_CONFIG: join(directory, 'config.json') },
  });
  assert.equal(child.status, 1);
  assert.match(child.stderr, /Explicit physical config cannot be null/);
} finally { rmSync(directory, { recursive: true, force: true }); }
const logits = Buffer.from(new Float32Array([0.25, -0.25, 0.5, 1]).buffer).toString('base64');
const reference = { kind: 'independent-source-model', modelIdentity: identity, planId: resolved.planId,
  provenance: { implementation: 'Independent test equations', implementationIdentity: identity },
  generation: resolved.generation, prompts: resolved.prompts,
  expected: [{ input: [1], steps: [
    { step: 0, context: [1], tokenId: 2, stopReason: null, logits },
    { step: 2, context: [1, 2, 3], tokenId: 0, stopReason: 'max-tokens', logits },
  ] }] };
const read = value => {
  const bytes = Buffer.from(JSON.stringify(value));
  return bindPhysicalIndependentReference(bytes, { sha256: physicalInputDigest(bytes) }, resolved);
};
assert.deepEqual([...read(reference).expected[0].steps[1].logits], [0.25, -0.25, 0.5, 1]);
assert.throws(() => bindPhysicalIndependentReference(Buffer.from(JSON.stringify(reference)),
  { sha256: 'sha256:' + '0'.repeat(64) }, resolved), /digest changed/);
for (const mutate of [
  value => { value.kind = 'doppler-parity-capture'; },
  value => { delete value.provenance; },
  value => { value.provenance.implementationIdentity = 'unpinned'; },
  value => { value.modelIdentity = 'sha256:' + '0'.repeat(64); },
  value => { value.planId = 'sha256:' + '0'.repeat(64); },
  value => { value.generation.seed = 18; },
  value => { value.prompts = ['Different prompt']; },
  value => { value.expected = []; },
  value => { value.expected[0].input = [4]; },
  value => { value.expected[0].steps[1].context = [1, 2]; },
  value => { value.expected[0].steps[1].context[0] = 0; },
  value => { value.expected[0].steps[1].context[1] = 0; },
  value => { value.expected[0].steps[0].step = 2; },
  value => { value.expected[0].steps[0].stopReason = 'eos-token'; },
  value => { value.expected[0].steps[0].tokenId = 4; },
  value => { value.expected[0].steps[0].logits += '!'; },
  value => { value.expected[0].steps[0].logits = Buffer.from(new Float32Array([1]).buffer).toString('base64'); },
  value => { value.expected[0].steps[0].logits = Buffer.from(new Float32Array([1, 2, NaN, 4]).buffer).toString('base64'); },
]) {
  const invalid = structuredClone(reference); mutate(invalid);
  assert.throws(() => read(invalid));
}
console.log('resident-partition-physical-config.test: ok');
