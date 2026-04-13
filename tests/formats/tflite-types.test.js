import assert from 'node:assert/strict';

import { buildGemma4LiteRTPackedFixture } from '../helpers/gemma4-litert-fixture.js';
import { buildTfliteFixture, FIXTURE_TFLITE_TENSOR_TYPE } from '../helpers/tflite-fixture.js';

const { parseTFLite } = await import('../../src/formats/tflite/types.js');

const embedBytes = Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7]);
const qProjBytes = Uint8Array.from([8, 9, 10, 11, 12, 13, 14, 15]);

const parsed = await parseTFLite(buildTfliteFixture({
  description: 'gemma4-tflite-fixture',
  tensors: [
    {
      name: 'model.embed_tokens.weight',
      shape: [2, 2],
      type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT16,
      data: embedBytes,
    },
    {
      name: 'model.layers.0.self_attn.q_proj.weight',
      shape: [2, 2],
      type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT16,
      data: qProjBytes,
    },
  ],
}));

assert.equal(parsed.schemaVersion, 3);
assert.equal(parsed.description, 'gemma4-tflite-fixture');
assert.equal(parsed.subgraphCount, 1);
assert.equal(parsed.mainSubgraphName, 'main');
assert.equal(parsed.sourceQuantization, 'F16');
assert.equal(parsed.tensors.length, 2);
assert.equal(parsed.tensors[0].dtype, 'F16');
assert.equal(parsed.tensors[0].offset >= 0, true);
assert.equal(parsed.tensors[0].size, embedBytes.byteLength);
assert.equal(parsed.tensors[1].name, 'model.layers.0.self_attn.q_proj.weight');
assert.equal(parsed.tensors[1].size, qProjBytes.byteLength);

const quantized = await parseTFLite(buildTfliteFixture({
  tensors: [{
    name: 'model.layers.0.self_attn.q_proj.weight',
    shape: [2, 2],
    type: FIXTURE_TFLITE_TENSOR_TYPE.INT8,
    data: Uint8Array.from([1, 2, 3, 4]),
    quantization: {
      scales: [0.25],
      zeroPoints: [1],
      quantizedDimension: 0,
    },
  }],
}));
assert.equal(quantized.sourceQuantization, 'F16');
assert.equal(quantized.tensors[0].dtype, 'F16');
assert.equal(quantized.tensors[0].sourceDtype, 'INT8');
assert.equal(quantized.tensors[0].sourceTransform?.kind, 'affine_dequant');
assert.equal(quantized.tensors[0].sourceTransform?.scale, 0.25);
assert.equal(quantized.tensors[0].sourceTransform?.zeroPoint, 1);

const externalBuffers = await parseTFLite(buildTfliteFixture({
  description: 'gemma4-external-buffer-fixture',
  externalBuffers: true,
  tensors: [
    {
      name: 'model.layers.0.self_attn.input_scale',
      shape: [1],
      type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT32,
      omitTypeField: true,
      data: Uint8Array.from([0, 0, 128, 63]),
    },
    {
      name: 'model.layers.0.self_attn.q_proj.weight',
      shape: [2, 2],
      type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT16,
      data: Uint8Array.from([9, 8, 7, 6, 5, 4, 3, 2]),
    },
  ],
}));
assert.equal(externalBuffers.description, 'gemma4-external-buffer-fixture');
assert.equal(externalBuffers.tensors.length, 2);
assert.equal(externalBuffers.tensors[0].dtype, 'F32');
assert.equal(externalBuffers.tensors[0].offset > 0, true);
assert.equal(externalBuffers.tensors[0].size, 4);
assert.equal(externalBuffers.tensors[1].dtype, 'F16');
assert.equal(externalBuffers.tensors[1].size, 8);

const packedLiteRT = await parseTFLite(
  buildGemma4LiteRTPackedFixture({ numLayers: 1 }),
  { allowPackedQuantization: true }
);
assert.equal(packedLiteRT.tensors.some((tensor) => tensor.name === 'transformer.layer_0.attn.q.w'), true);
assert.equal(
  packedLiteRT.tensors.find((tensor) => tensor.name === 'transformer.layer_0.attn.q.w')?.dtype,
  'INT4'
);
assert.equal(
  packedLiteRT.tensors.find((tensor) => tensor.name === 'transformer.layer_0.per_layer_embeddings.w')?.dtype,
  'UINT8'
);
assert.equal(
  packedLiteRT.tensors.find((tensor) => tensor.name === 'transformer.layer_0.attn.q.w')?.sourceTransform ?? null,
  null
);

await assert.rejects(
  () => parseTFLite(buildTfliteFixture({
    tensors: [{
      name: 'transformer.layer_0.attn.q.w',
      shape: [8],
      type: FIXTURE_TFLITE_TENSOR_TYPE.INT4,
      data: Uint8Array.from([1, 2, 3, 4]),
    }],
  })),
  /Packed LiteRT-LM companion tensors/
);

await assert.rejects(
  () => parseTFLite(buildTfliteFixture({
    tensors: [{
      name: 'model.layers.0.self_attn.q_proj.weight',
      shape: [2, 2],
      type: FIXTURE_TFLITE_TENSOR_TYPE.INT8,
      data: Uint8Array.from([1, 2, 3, 4]),
      quantization: {
        scales: [0.5, 0.25],
        zeroPoints: [0, 0],
        quantizedDimension: 1,
      },
    }],
  })),
  /Per-channel quantization is not supported/
);

console.log('tflite-types.test: ok');
