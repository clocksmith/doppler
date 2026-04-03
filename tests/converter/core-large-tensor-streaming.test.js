import assert from 'node:assert/strict';

import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { convertModel } = await import('../../src/converter/core.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

const model = {
  name: 'large-tensor-streaming-test',
  modelId: 'large-tensor-streaming-test',
  quantization: 'BF16',
  tensors: [
    {
      name: 'model.layers.0.self_attn.q_proj.weight',
      shape: [4097, 524288],
      dtype: 'BF16',
      size: 4296015872,
      offset: 0,
      sourcePath: '/tmp/fake-large.safetensors',
    },
  ],
  config: {
    model_type: 'gemma2',
  },
};

const baseOptions = {
  modelId: 'large-tensor-streaming-test',
  modelType: 'transformer',
  quantization: 'F16',
  quantizationInfo: {
    weights: 'f16',
    embeddings: 'f16',
    compute: 'f16',
    variantTag: 'f16',
  },
  architecture: {
    numLayers: 1,
    hiddenSize: 2,
    intermediateSize: 8,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 2,
    vocabSize: 16,
    maxSeqLen: 8,
    ropeTheta: 1000000,
  },
  inference: {
    ...DEFAULT_MANIFEST_INFERENCE,
  },
  eosTokenId: 1,
  converterConfig: createConverterConfig(),
};

{
  let readTensorDataCalled = false;
  const writtenShards = [];
  let capturedManifest = null;
  const io = {
    async readTensorData() {
      readTensorDataCalled = true;
      throw new Error('readTensorData should not be called for oversized streamed tensors');
    },
    async writeShard(index, data) {
      writtenShards.push({
        index,
        bytes: Array.from(data),
      });
      return `hash-${index}`;
    },
    async writeManifest(manifest) {
      capturedManifest = manifest;
    },
  };

  const result = await convertModel(model, io, {
    ...baseOptions,
    shardSize: 4,
    async largeTensorTransformer(input) {
      assert.equal(input.tensor.name, 'model.layers.0.self_attn.q_proj.weight');
      await input.writeChunk({
        tensorData: new Uint8Array([1, 2, 3]),
        outDtype: 'F16',
        outLayout: 'row',
      });
      await input.writeChunk({
        tensorData: new Uint8Array([4, 5]),
        outDtype: 'F16',
        outLayout: 'row',
      });
      input.reportProgress?.(4296015872, 4296015872);
      return {
        outDtype: 'F16',
        outLayout: 'row',
      };
    },
  });

  assert.equal(readTensorDataCalled, false);
  assert.equal(result.shardCount, 2);
  assert.equal(result.totalSize, 5);
  assert.deepEqual(writtenShards, [
    { index: 0, bytes: [1, 2, 3, 4] },
    { index: 1, bytes: [5] },
  ]);
  assert.ok(capturedManifest, 'manifest should be written');
  assert.deepEqual(
    capturedManifest.tensors['model.layers.0.self_attn.q_proj.weight'].spans,
    [
      { shardIndex: 0, offset: 0, size: 3 },
      { shardIndex: 0, offset: 3, size: 1 },
      { shardIndex: 1, offset: 0, size: 1 },
    ]
  );
  assert.equal(capturedManifest.tensors['model.layers.0.self_attn.q_proj.weight'].size, 5);
  assert.equal(capturedManifest.tensors['model.layers.0.self_attn.q_proj.weight'].dtype, 'F16');
  assert.equal(capturedManifest.tensors['model.layers.0.self_attn.q_proj.weight'].layout, 'row');
}

{
  const io = {
    async readTensorData() {
      throw new Error('readTensorData should not be called before the oversized tensor check');
    },
    async writeShard() {
      throw new Error('writeShard should not run on oversized tensor failure');
    },
    async writeManifest() {
      throw new Error('writeManifest should not run on oversized tensor failure');
    },
  };

  await assert.rejects(
    () => convertModel(model, io, {
      ...baseOptions,
      shardSize: 4,
    }),
    /Provide a largeTensorTransformer for streamed conversion/
  );
}

console.log('core-large-tensor-streaming.test: ok');
