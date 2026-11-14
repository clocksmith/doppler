import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createOrderedLayerPartitionPlanFromManifest, validateOrderedLayerPartitionPlan,
  hashLayerPartitionPlan, getPartitionActivationContract } from '../../src/config/layer-partition-plan.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

const manifest = { modelId: 'wire-fixture', architecture: { numLayers: 4, hiddenSize: 8, vocabSize: 16 } };
const request = { splitLayers: [2], activationDtype: 'f16', activationPrecision: 'f16' };
test('execution precision is explicit, distinct from wire layout and digest-bound', () => {
  const ordinary = createOrderedLayerPartitionPlanFromManifest(manifest, request);
  const cast = createOrderedLayerPartitionPlanFromManifest(manifest, { ...request, executionPrecision: 'f32' });
  assert.equal(Object.hasOwn(ordinary, 'executionPrecision'), false);
  assert.equal(cast.executionPrecision, 'f32');
  assert.deepEqual(cast.partitions, ordinary.partitions);
  assert.equal(getPartitionActivationContract(cast).elementStrideBytes, 2);
  assert.notEqual(hashLayerPartitionPlan(cast), hashLayerPartitionPlan(ordinary));
  assert.equal(hashLayerPartitionPlan(ordinary), computeCanonicalSha256(ordinary));
  assert.deepEqual(validateOrderedLayerPartitionPlan(manifest, JSON.parse(JSON.stringify(cast))), cast);
  assert.throws(() => validateOrderedLayerPartitionPlan(manifest, { ...cast, executionPrecision: 'f16' }), /execution precision/);
});
test('hyperconnection streams may cross f16 boundaries only with explicit f32 execution', () => {
  const hyper = { ...manifest, inference: { hyperConnections: { streamCount: 4, streamWidth: 8 } } };
  assert.throws(() => createOrderedLayerPartitionPlanFromManifest(hyper, request), /scalar f32/);
  const cast = createOrderedLayerPartitionPlanFromManifest(hyper, { ...request, executionPrecision: 'f32' });
  assert.equal(cast.partitions[0].outputContract.hiddenSize, 32);
  assert.equal(cast.partitions[0].outputContract.dtype, 'f16');
  assert.deepEqual(validateOrderedLayerPartitionPlan(hyper, cast), cast);
  assert.ok(hashLayerPartitionPlan(cast).startsWith('sha256:'));
});
test('rejects unsupported or ambiguous execution/wire precision combinations', () => {
  for (const extra of [
    { executionPrecision: null }, { executionPrecision: 'f16' }, { executionPrecision: 'two-f32' },
    { executionPrecision: 'f32', activationDtype: 'f32', activationPrecision: 'f32' },
    { executionPrecision: 'f32', activationDtype: 'f32', activationPrecision: 'two-f32' },
    { executionPrecision: 'f32', activationPrecision: undefined },
  ]) assert.throws(() => createOrderedLayerPartitionPlanFromManifest(manifest, { ...request, ...extra }));
});
