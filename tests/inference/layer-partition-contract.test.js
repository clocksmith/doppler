import assert from 'node:assert/strict';

import {
  LAYER_PARTITION_SCHEMA,
  ACTIVATION_TENSOR_SCHEMA,
  PARTITION_COMPARISON_SCHEMA,
  createLayerPartitionPlan,
  validateActivationTensorShape,
  serializeActivationFrame,
  deserializeActivationFrame,
  createPartitionContinuation,
  comparePartitionExecution
} from '../../src/inference/pipelines/text/layer-partition-contract.js';

// 1. Partition Plan Creation
{
  const plan = createLayerPartitionPlan({
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    numLayers: 24,
    hiddenSize: 1024,
    vocabSize: 248320,
    splitLayer: 12
  });

  assert.equal(plan.schema, LAYER_PARTITION_SCHEMA);
  assert.equal(plan.totalLayers, 24);
  assert.equal(plan.splitLayer, 12);
  assert.equal(plan.partitions.length, 2);

  // Group 0: Device A
  const p0 = plan.partitions[0];
  assert.equal(p0.index, 0);
  assert.deepEqual(p0.layerRange, [0, 11]);
  assert.equal(p0.hasEmbedding, true);
  assert.equal(p0.hasLmHead, false);
  assert.equal(p0.inputContract.type, 'token-ids');
  assert.equal(p0.outputContract.type, 'activation-tensor');
  assert.equal(p0.outputContract.hiddenSize, 1024);

  // Group 1: Device B
  const p1 = plan.partitions[1];
  assert.equal(p1.index, 1);
  assert.deepEqual(p1.layerRange, [12, 23]);
  assert.equal(p1.hasEmbedding, false);
  assert.equal(p1.hasLmHead, true);
  assert.equal(p1.inputContract.type, 'activation-tensor');
  assert.equal(p1.outputContract.type, 'logits');
  assert.equal(p1.outputContract.vocabSize, 248320);

  // Validation on invalid inputs
  assert.throws(() => createLayerPartitionPlan({ modelId: '', numLayers: 24, hiddenSize: 1024, vocabSize: 100 }), /modelId/);
  assert.throws(() => createLayerPartitionPlan({ modelId: 'test', numLayers: 1, hiddenSize: 1024, vocabSize: 100 }), /numLayers/);
  assert.throws(() => createLayerPartitionPlan({ modelId: 'test', numLayers: 24, hiddenSize: 1024, vocabSize: 100, splitLayer: 24 }), /splitLayer/);
  assert.throws(() => createLayerPartitionPlan({ modelId: 'test', numLayers: 24, hiddenSize: 1024, vocabSize: 100, splitLayer: 0 }), /splitLayer/);
}

// 2. Activation Tensor Serialization and Deserialization
{
  const shape = [1, 4, 8];
  const totalElements = 1 * 4 * 8;
  const floatData = new Float32Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    floatData[i] = (i + 1) * 0.1;
  }

  const frame = serializeActivationFrame({
    shape,
    dtype: 'f32',
    data: floatData,
    seqOffset: 0,
    step: 1,
    metadata: { modelId: 'test-model' }
  });

  assert.equal(frame.schema, ACTIVATION_TENSOR_SCHEMA);
  assert.equal(frame.byteLength, totalElements * 4);
  assert.deepEqual(frame.shape, shape);
  assert.equal(frame.step, 1);

  const deserialized = deserializeActivationFrame(frame);
  assert.deepEqual(deserialized.shape, shape);
  assert.equal(deserialized.dtype, 'f32');
  assert.equal(deserialized.step, 1);
  assert.equal(deserialized.tensorData.length, totalElements);
  assert.equal(deserialized.tensorData[0], floatData[0]);
  assert.equal(deserialized.tensorData[totalElements - 1], floatData[totalElements - 1]);
}

// 3. Partition Continuation State Isolation
{
  const cont0 = createPartitionContinuation({
    partitionIndex: 0,
    totalLayers: 24,
    layerRange: [0, 11]
  });
  const cont1 = createPartitionContinuation({
    partitionIndex: 1,
    totalLayers: 24,
    layerRange: [12, 23]
  });

  cont0.setLayerKVCache(5, { k: 'mock-k-5', v: 'mock-v-5' });
  assert.deepEqual(cont0.getLayerKVCache(5), { k: 'mock-k-5', v: 'mock-v-5' });
  assert.equal(cont0.getLayerKVCache(6), null);
  assert.throws(() => cont0.getLayerKVCache(15), /outside partition 0/);
  assert.throws(() => cont1.getLayerKVCache(5), /outside partition 1/);

  cont0.advance(4);
  assert.equal(cont0.getSequenceOffset(), 4);
  cont0.reset();
  assert.equal(cont0.getSequenceOffset(), 0);
  assert.equal(cont0.getLayerKVCache(5), null);
}

// 4. Numerical Comparison Contract
{
  const ref = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
  const exact = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
  const slight = new Float32Array([1.00002, 2.00001, 2.99998, 4.00003, 4.99999]);
  const divergent = new Float32Array([1.1, 2.0, 3.0, 4.0, 5.0]);

  const exactComparison = comparePartitionExecution({
    splitOutput: exact,
    referenceOutput: ref,
    tolerance: 1e-4
  });
  assert.equal(exactComparison.schema, PARTITION_COMPARISON_SCHEMA);
  assert.equal(exactComparison.matches, true);
  assert.equal(exactComparison.maxDiff, 0);
  assert.equal(exactComparison.cosineSimilarity, 1);

  const slightComparison = comparePartitionExecution({
    splitOutput: slight,
    referenceOutput: ref,
    tolerance: 1e-4
  });
  assert.equal(slightComparison.matches, true);
  assert.ok(slightComparison.maxDiff <= 1e-4);
  assert.ok(slightComparison.cosineSimilarity >= 0.9999);

  const divergentComparison = comparePartitionExecution({
    splitOutput: divergent,
    referenceOutput: ref,
    tolerance: 1e-4
  });
  assert.equal(divergentComparison.matches, false);
  assert.ok(divergentComparison.maxDiff > 1e-4);
}
