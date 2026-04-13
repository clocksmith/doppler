import { buildTfliteFixture, FIXTURE_TFLITE_TENSOR_TYPE } from './tflite-fixture.js';

function encodeFloat32(values) {
  return new Uint8Array(Float32Array.from(values).buffer);
}

function encodeRepeatedFloat32(length, value) {
  return new Uint8Array(Float32Array.from({ length }, () => value).buffer);
}

function encodeRepeatedInt32(length, value) {
  return new Uint8Array(Int32Array.from({ length }, () => value).buffer);
}

function pushFloatTensor(tensors, name, values) {
  tensors.push({
    name,
    shape: [values.length],
    type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT32,
    data: encodeFloat32(values),
  });
}

function pushRowwiseTensor(tensors, name, type, dataBytes, rowScales, rowSums = null) {
  tensors.push({
    name,
    shape: [dataBytes.byteLength],
    type,
    data: dataBytes,
  });
  tensors.push({
    name: `${name}_quantized_scale`,
    shape: [rowScales.length],
    type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT32,
    data: encodeFloat32(rowScales),
  });
  if (Array.isArray(rowSums) || ArrayBuffer.isView(rowSums)) {
    tensors.push({
      name: `${name}.sum_i`,
      shape: [rowSums.length],
      type: FIXTURE_TFLITE_TENSOR_TYPE.INT32,
      data: new Uint8Array(Int32Array.from(rowSums).buffer),
    });
  }
}

export function buildGemma4LiteRTPackedFixture(options = {}) {
  if (options.profileAligned === true) {
    const tensors = [];
    pushRowwiseTensor(
      tensors,
      'transformer.embedder.input_embedding.w',
      FIXTURE_TFLITE_TENSOR_TYPE.INT8,
      new Uint8Array(262144 * 1536 / 4),
      new Float32Array(262144).fill(0.25),
      new Int32Array(262144).fill(0)
    );
    pushRowwiseTensor(
      tensors,
      'transformer.embedder.per_layer_model_projection.w',
      FIXTURE_TFLITE_TENSOR_TYPE.INT8,
      new Uint8Array(8960 * 1536),
      new Float32Array(8960).fill(0.25),
      new Int32Array(8960).fill(0)
    );
    tensors.push({
      name: 'transformer.embedder.per_layer_projection_norm.scale',
      shape: [256],
      type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT32,
      data: encodeRepeatedFloat32(256, 1),
    });
    tensors.push({
      name: 'transformer.final_norm.scale',
      shape: [1536],
      type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT32,
      data: encodeRepeatedFloat32(1536, 1),
    });
    pushRowwiseTensor(
      tensors,
      'transformer.layer_0.attn.q.w',
      FIXTURE_TFLITE_TENSOR_TYPE.INT4,
      new Uint8Array((2048 * 1536) / 2),
      new Float32Array(2048).fill(0.25),
      new Int32Array(2048).fill(0)
    );
    pushRowwiseTensor(
      tensors,
      'transformer.layer_0.per_layer_embedding_gate.w',
      FIXTURE_TFLITE_TENSOR_TYPE.INT8,
      new Uint8Array(256 * 1536),
      new Float32Array(256).fill(0.25),
      new Int32Array(256).fill(0)
    );
    pushRowwiseTensor(
      tensors,
      'transformer.layer_34.per_layer_embeddings.w',
      FIXTURE_TFLITE_TENSOR_TYPE.UINT8,
      new Uint8Array((262144 * 256) / 2),
      new Float32Array(262144).fill(0.25)
    );
    return buildTfliteFixture({
      description: options.description || 'gemma4-litert-profile-aligned-fixture',
      tensors,
    });
  }

  const numLayers = Number.isInteger(options.numLayers) && options.numLayers > 0
    ? options.numLayers
    : 35;
  const rowScales = [0.25, 0.5];
  const rowSums = [0, 0];
  const int8Bytes = Uint8Array.from([1, 2, 3, 4]);
  const int4Bytes = Uint8Array.from([0x98, 0xba]);
  const perLayerEmbeddingBytes = Uint8Array.from([0x87, 0xa9]);
  const tensors = [];

  pushRowwiseTensor(
    tensors,
    'transformer.embedder.input_embedding.w',
    FIXTURE_TFLITE_TENSOR_TYPE.INT8,
    int8Bytes,
    rowScales,
    rowSums
  );
  pushRowwiseTensor(
    tensors,
    'transformer.embedder.per_layer_model_projection.w',
    FIXTURE_TFLITE_TENSOR_TYPE.INT8,
    int8Bytes,
    rowScales,
    rowSums
  );
  pushFloatTensor(
    tensors,
    'transformer.embedder.per_layer_projection_norm.scale',
    [1, 1]
  );
  pushFloatTensor(
    tensors,
    'transformer.final_norm.scale',
    [1, 1]
  );

  for (let layerIndex = 0; layerIndex < numLayers; layerIndex += 1) {
    const prefix = `transformer.layer_${layerIndex}`;
    pushFloatTensor(tensors, `${prefix}.pre_attention_norm.scale`, [1, 1]);
    pushRowwiseTensor(tensors, `${prefix}.attn.q.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT4, int4Bytes, rowScales, rowSums);
    pushFloatTensor(tensors, `${prefix}.attn.q_norm.scale`, [1, 1]);
    pushRowwiseTensor(tensors, `${prefix}.attn.k.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT4, int4Bytes, rowScales, rowSums);
    pushRowwiseTensor(tensors, `${prefix}.attn.v.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT4, int4Bytes, rowScales, rowSums);
    pushFloatTensor(tensors, `${prefix}.attn.k_norm.scale`, [1, 1]);
    pushRowwiseTensor(tensors, `${prefix}.attn.attn_vec_einsum.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT4, int4Bytes, rowScales, rowSums);
    pushFloatTensor(tensors, `${prefix}.post_attention_norm.scale`, [1, 1]);
    pushFloatTensor(tensors, `${prefix}.pre_ffw_norm.scale`, [1, 1]);
    pushFloatTensor(tensors, `${prefix}.post_ffw_norm.scale`, [1, 1]);
    pushFloatTensor(tensors, `${prefix}.post_per_layer_input_norm.scale`, [1, 1]);
    pushRowwiseTensor(tensors, `${prefix}.mlp.ff_gate.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT4, int4Bytes, rowScales, rowSums);
    pushRowwiseTensor(tensors, `${prefix}.mlp.ff1.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT4, int4Bytes, rowScales, rowSums);
    pushRowwiseTensor(tensors, `${prefix}.mlp.linear.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT4, int4Bytes, rowScales, rowSums);
    pushRowwiseTensor(tensors, `${prefix}.per_layer_embedding_gate.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT8, int8Bytes, rowScales, rowSums);
    pushRowwiseTensor(tensors, `${prefix}.per_layer_embedding_projection.w`, FIXTURE_TFLITE_TENSOR_TYPE.INT8, int8Bytes, rowScales, rowSums);
    pushRowwiseTensor(
      tensors,
      `${prefix}.per_layer_embeddings.w`,
      FIXTURE_TFLITE_TENSOR_TYPE.UINT8,
      perLayerEmbeddingBytes,
      rowScales
    );
  }

  return buildTfliteFixture({
    description: options.description || 'gemma4-litert-packed-fixture',
    tensors,
  });
}
