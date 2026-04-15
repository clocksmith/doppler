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

function encodeUint8(values) {
  if (values instanceof Uint8Array) {
    return values;
  }
  return Uint8Array.from(Array.isArray(values) ? values : [...values], (value) => Number(value) & 0xff);
}

function computeShapeElementCount(shape) {
  if (!Array.isArray(shape) || shape.length === 0) {
    return 0;
  }
  return shape.reduce((acc, value) => acc * Math.max(0, Number(value) || 0), 1);
}

function repeatUint8Pattern(patternBytes, targetLength) {
  if (!(patternBytes instanceof Uint8Array)) {
    throw new Error('gemma4-litert-fixture: scale companion pattern must be Uint8Array.');
  }
  if (!Number.isInteger(targetLength) || targetLength <= 0) {
    return new Uint8Array(0);
  }
  if (patternBytes.length === 0) {
    throw new Error('gemma4-litert-fixture: scale companion pattern cannot be empty.');
  }
  const bytes = new Uint8Array(targetLength);
  for (let index = 0; index < targetLength; index += 1) {
    bytes[index] = patternBytes[index % patternBytes.length];
  }
  return bytes;
}

function encodeUtf8(value) {
  return new TextEncoder().encode(String(value ?? ''));
}

function pushFloatTensor(tensors, name, values) {
  tensors.push({
    name,
    shape: [values.length],
    type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT32,
    data: encodeFloat32(values),
  });
}

function pushRowwiseTensor(tensors, name, type, dataBytes, rowScales, rowSums = null, options = {}) {
  const scaleCompanionType = Number.isInteger(options.scaleCompanionType)
    ? options.scaleCompanionType
    : FIXTURE_TFLITE_TENSOR_TYPE.FLOAT32;
  const scaleElementCount = Array.isArray(rowScales)
    ? rowScales.length
    : rowScales instanceof Uint8Array || rowScales instanceof Float32Array || rowScales instanceof Int32Array
      ? rowScales.length
      : 0;
  const scaleShape = Array.isArray(options.scaleCompanionShape)
    ? options.scaleCompanionShape
    : (
      scaleCompanionType === FIXTURE_TFLITE_TENSOR_TYPE.UINT8 && scaleElementCount > 0 && scaleElementCount % 4 === 0
        ? [scaleElementCount / 4, 4]
        : [scaleElementCount]
    );
  const scaleBytes = scaleCompanionType === FIXTURE_TFLITE_TENSOR_TYPE.UINT8
    ? (() => {
      const explicitScaleBytes = options.scaleCompanionData instanceof Uint8Array
        ? options.scaleCompanionData
        : encodeUint8(rowScales);
      const requiredScaleElements = computeShapeElementCount(scaleShape);
      if (!Number.isInteger(requiredScaleElements) || requiredScaleElements < 0) {
        return explicitScaleBytes;
      }
      if (requiredScaleElements === explicitScaleBytes.length) {
        return explicitScaleBytes;
      }
      if (
        requiredScaleElements < explicitScaleBytes.length
        || explicitScaleBytes.length === 0
        || requiredScaleElements % explicitScaleBytes.length !== 0
      ) {
        throw new Error(
          `gemma4-litert-fixture: UINT8 scale companion bytes (${explicitScaleBytes.length}) cannot target shape ${JSON.stringify(scaleShape)}.`
        );
      }
      return repeatUint8Pattern(explicitScaleBytes, requiredScaleElements);
    })()
    : encodeFloat32(rowScales);
  tensors.push({
    name,
    shape: [dataBytes.byteLength],
    type,
    data: dataBytes,
  });
  tensors.push({
    name: `${name}_quantized_scale`,
    shape: scaleShape,
    type: scaleCompanionType,
    data: scaleBytes,
    ...(scaleCompanionType === FIXTURE_TFLITE_TENSOR_TYPE.UINT8 && options.scaleCompanionQuantization
      ? {
        quantization: options.scaleCompanionQuantization,
      }
      : {}),
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
    pushFloatTensor(
      tensors,
      'transformer.embedder.per_layer_model_projection.input_activation_static_scale',
      [0.25]
    );
    pushFloatTensor(
      tensors,
      'transformer.embedder.per_layer_model_projection.output_activation_static_scale',
      [0.125]
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
      'transformer.layer_0.attn.attn_vec_einsum.w',
      FIXTURE_TFLITE_TENSOR_TYPE.INT4,
      new Uint8Array((1536 * 2048) / 2),
      new Float32Array(1536).fill(0.25),
      new Int32Array(1536).fill(0)
    );
    pushRowwiseTensor(
      tensors,
      'transformer.layer_0.mlp.linear.w',
      FIXTURE_TFLITE_TENSOR_TYPE.INT4,
      new Uint8Array((1536 * 6144) / 2),
      new Float32Array(1536).fill(0.25),
      new Int32Array(1536).fill(0)
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
      'transformer.layer_0.per_layer_embedding_projection.w',
      FIXTURE_TFLITE_TENSOR_TYPE.INT8,
      new Uint8Array(1536 * 256),
      new Float32Array(1536).fill(0.25),
      new Int32Array(1536).fill(0)
    );
    pushRowwiseTensor(
      tensors,
      'transformer.layer_34.per_layer_embeddings.w',
      FIXTURE_TFLITE_TENSOR_TYPE.UINT8,
      new Uint8Array((262144 * 256) / 2),
      Uint8Array.from([1, 2, 3, 4]),
      null,
      {
        scaleCompanionType: FIXTURE_TFLITE_TENSOR_TYPE.UINT8,
        scaleCompanionShape: [262144, 4],
        scaleCompanionQuantization: {
          scales: [0.01],
          zeroPoints: [0],
        },
        scaleCompanionData: repeatUint8Pattern(Uint8Array.from([1, 2, 3, 4]), 262144 * 4),
      }
    );
    return buildTfliteFixture({
      description: options.description || 'gemma4-litert-profile-aligned-fixture',
      tensors,
      metadata: [
        {
          name: 'odml.infra.proto.LlmParameters',
          data: Uint8Array.from([0x08, 0x01, 0x10, 0x02]),
        },
        {
          name: 'spm_vocab_model',
          data: encodeUtf8('<pad><eos><bos>'),
        },
        {
          name: 'backend',
          data: encodeUtf8('gpu'),
        },
      ],
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
    'transformer.embedder.per_layer_model_projection.input_activation_static_scale',
    [0.25]
  );
  pushFloatTensor(
    tensors,
    'transformer.embedder.per_layer_model_projection.output_activation_static_scale',
    [0.125]
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
      Uint8Array.from([1, 2, 3, 4]),
      null,
      {
        scaleCompanionType: FIXTURE_TFLITE_TENSOR_TYPE.UINT8,
        scaleCompanionShape: [262144, 4],
        scaleCompanionData: repeatUint8Pattern(Uint8Array.from([1, 2, 3, 4]), 262144 * 4),
        scaleCompanionQuantization: {
          scales: [0.01],
          zeroPoints: [0],
        },
      }
    );
  }

  return buildTfliteFixture({
    description: options.description || 'gemma4-litert-packed-fixture',
    tensors,
    metadata: [
      {
        name: 'odml.infra.proto.LlmParameters',
        data: Uint8Array.from([0x08, 0x01, 0x10, 0x02]),
      },
      {
        name: 'spm_vocab_model',
        data: encodeUtf8('<pad><eos><bos>'),
      },
      {
        name: 'backend',
        data: encodeUtf8('gpu'),
      },
    ],
  });
}
