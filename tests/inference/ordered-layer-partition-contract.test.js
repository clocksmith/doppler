import assert from 'node:assert/strict';
import * as contract from '../../src/partitions.js';
const dimensions = { modelId: 'ordered-fixture', numLayers: 28, hiddenSize: 4, vocabSize: 16 };
const manifest = { modelId: dimensions.modelId, architecture: dimensions };
assert.equal(typeof contract.createOrderedLayerPartitionPlan, 'function', 'The ordered plan constructor must be public.');
for (const [stages, splits] of [[2, [14]], [3, [3, 19]], [4, [1, 8, 24]], [7, [1, 3, 8, 14, 18, 27]]]) {
  for (const precision of ['f32', 'f16', 'two-f32']) {
    const dtype = precision === 'f16' ? 'f16' : 'f32';
    const plan = contract.createOrderedLayerPartitionPlan({ ...dimensions, splitLayers: splits, activationDtype: dtype, activationPrecision: precision });
    assert.equal(plan.schema, 'doppler.layer-partition-contract/v3');
    assert.equal(plan.stageCount, stages);
    assert.deepEqual(contract.validateOrderedLayerPartitionPlan(manifest, JSON.parse(JSON.stringify(plan))), plan);
    const visited = [];
    for (const [i, stage] of plan.partitions.entries()) {
      assert.equal(stage.index, i);
      assert.equal(stage.hasEmbedding, i === 0);
      assert.equal(stage.hasLmHead, i === stages - 1);
      assert.equal(stage.inputContract.type, i === 0 ? 'token-ids' : 'activation-tensor');
      assert.equal(stage.outputContract.type, i === stages - 1 ? 'logits' : 'activation-tensor');
      assert.equal(stage.layerCount, stage.layerRange[1] - stage.layerRange[0] + 1);
      for (let layer = stage.layerRange[0]; layer <= stage.layerRange[1]; layer++) visited.push(layer);
      if (i) assert.deepEqual(stage.inputContract, plan.partitions[i - 1].outputContract);
      assert(Object.isFrozen(stage.layerRange) && Object.isFrozen(stage.inputContract));
    }
    assert.deepEqual(visited, Array.from({ length: 28 }, (_, i) => i));
    assert(Object.isFrozen(plan.splitLayers) && Object.isFrozen(plan.partitions));
    const layout = contract.getPartitionActivationContract(plan);
    const bytes = new Uint8Array(4 * layout.elementStrideBytes);
    bytes[0] = 1; bytes[bytes.length - 1] = 128;
    const frame = contract.serializeActivationFrame({ ...layout, shape: [1, 1, 4], data: bytes });
    assert.deepEqual(new Uint8Array(contract.validateActivationFrameForPlan(frame, plan).tensorData.buffer), bytes);
    const mutations = [p => p.partitions.reverse(), p => p.partitions[0].layerRange[1]++,
      p => p.partitions[1].hasEmbedding = true, p => p.partitions[0].hasLmHead = true,
      p => p.partitions[1].inputContract.hiddenSize++, p => p.stageCount++, p => p.activationElementStrideBytes++,
      p => p.splitLayers.push(27), p => p.extra = 'unknown'];
    for (const mutate of mutations) {
      const bad = structuredClone(plan); mutate(bad);
      assert.throws(() => contract.validateOrderedLayerPartitionPlan(manifest, bad));
      assert.throws(() => contract.hashLayerPartitionPlan(bad));
    }
    assert.throws(() => contract.validateOrderedLayerPartitionPlan({ ...manifest, modelId: 'other' }, plan));
    assert.throws(() => contract.resolveLayerPartition(manifest, { plan, index: 0 }), /allocation/);
  }
}
for (const splitLayers of [[], [0], [28], [4, 4], [14, 3], [1.5], [NaN], [Infinity], null, '14']) {
  assert.throws(() => contract.createOrderedLayerPartitionPlan({ ...dimensions, splitLayers }));
}
for (const [activationPrecision, expected] of [[undefined, 'sha256:9fe509881c9a93f0894b3f8b01e2850ff79bff94bb01ea434c0aae9e2ec05e38'],
  ['two-f32', 'sha256:da46dcc12d154ba125c4321427943affd439aac8dcd2e4ee0e93ec3bcc585a7a']]) {
  assert.equal(contract.hashLayerPartitionPlan(contract.createLayerPartitionPlan({ ...dimensions, splitLayer: 14, activationPrecision })), expected);
}
console.log('ordered layer partition contracts passed: 2/3/4/7 stages, scalar/paired layouts, mutation rejection and legacy identities');
