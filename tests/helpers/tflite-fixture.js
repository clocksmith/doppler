import { Builder } from 'flatbuffers';

export const FIXTURE_TFLITE_TENSOR_TYPE = Object.freeze({
  FLOAT32: 0,
  FLOAT16: 1,
  INT32: 2,
  UINT8: 3,
  INT64: 4,
  STRING: 5,
  BOOL: 6,
  INT16: 7,
  COMPLEX64: 8,
  INT8: 9,
  FLOAT64: 10,
  COMPLEX128: 11,
  UINT64: 12,
  RESOURCE: 13,
  VARIANT: 14,
  UINT32: 15,
  UINT16: 16,
  INT4: 17,
  BFLOAT16: 18,
});

function createInt32Vector(builder, values) {
  builder.startVector(4, values.length, 4);
  for (let index = values.length - 1; index >= 0; index--) {
    builder.addInt32(values[index]);
  }
  return builder.endVector();
}

function createFloat32Vector(builder, values) {
  builder.startVector(4, values.length, 4);
  for (let index = values.length - 1; index >= 0; index--) {
    builder.addFloat32(values[index]);
  }
  return builder.endVector();
}

function createInt64Vector(builder, values) {
  builder.startVector(8, values.length, 8);
  for (let index = values.length - 1; index >= 0; index--) {
    builder.addInt64(BigInt(values[index]));
  }
  return builder.endVector();
}

function createOffsetVector(builder, offsets) {
  builder.startVector(4, offsets.length, 4);
  for (let index = offsets.length - 1; index >= 0; index--) {
    builder.addOffset(offsets[index]);
  }
  return builder.endVector();
}

function createBuffer(builder, data, options = {}) {
  const dataOffset = data && data.byteLength > 0 ? builder.createByteVector(data) : 0;
  const external = options.external === true;
  builder.startObject(external ? 3 : 1);
  if (external) {
    builder.addFieldInt64(1, BigInt(1), BigInt(0));
    builder.addFieldInt64(2, BigInt(Math.max(0, data?.byteLength ?? 0)), BigInt(0));
  } else if (dataOffset) {
    builder.addFieldOffset(0, dataOffset, 0);
  }
  return builder.endObject();
}

function createQuantizationParameters(builder, options) {
  const scales = Array.isArray(options?.scales) ? options.scales : [];
  const zeroPoints = Array.isArray(options?.zeroPoints) ? options.zeroPoints : [];
  const scalesOffset = scales.length > 0 ? createFloat32Vector(builder, scales) : 0;
  const zeroPointsOffset = zeroPoints.length > 0 ? createInt64Vector(builder, zeroPoints) : 0;
  builder.startObject(6);
  if (scalesOffset) {
    builder.addFieldOffset(2, scalesOffset, 0);
  }
  if (zeroPointsOffset) {
    builder.addFieldOffset(3, zeroPointsOffset, 0);
  }
  if (Number.isFinite(options?.quantizedDimension)) {
    builder.addFieldInt32(5, Math.trunc(options.quantizedDimension), 0);
  }
  return builder.endObject();
}

function createTensor(builder, options) {
  const shapeOffset = createInt32Vector(builder, options.shape);
  const nameOffset = builder.createString(options.name);
  const quantizationOffset = options.quantization
    ? createQuantizationParameters(builder, options.quantization)
    : 0;
  builder.startObject(8);
  builder.addFieldOffset(0, shapeOffset, 0);
  if (!(options.omitTypeField === true && options.type === FIXTURE_TFLITE_TENSOR_TYPE.FLOAT32)) {
    builder.addFieldInt8(1, options.type, 0);
  }
  builder.addFieldInt32(2, options.buffer, 0);
  builder.addFieldOffset(3, nameOffset, 0);
  if (quantizationOffset) {
    builder.addFieldOffset(4, quantizationOffset, 0);
  }
  if (options.isVariable === true) {
    builder.addFieldInt8(5, 1, 0);
  }
  return builder.endObject();
}

function createSubGraph(builder, options) {
  const tensorsOffset = createOffsetVector(builder, options.tensorOffsets);
  const inputsOffset = options.inputs.length > 0 ? createInt32Vector(builder, options.inputs) : 0;
  const outputsOffset = options.outputs.length > 0 ? createInt32Vector(builder, options.outputs) : 0;
  const nameOffset = builder.createString(options.name);
  builder.startObject(5);
  builder.addFieldOffset(0, tensorsOffset, 0);
  if (inputsOffset) {
    builder.addFieldOffset(1, inputsOffset, 0);
  }
  if (outputsOffset) {
    builder.addFieldOffset(2, outputsOffset, 0);
  }
  builder.addFieldOffset(4, nameOffset, 0);
  return builder.endObject();
}

function createModel(builder, options) {
  const subgraphsOffset = createOffsetVector(builder, [options.subgraphOffset]);
  const buffersOffset = createOffsetVector(builder, options.bufferOffsets);
  const descriptionOffset = builder.createString(options.description);
  builder.startObject(8);
  builder.addFieldInt32(0, options.version, 0);
  builder.addFieldOffset(2, subgraphsOffset, 0);
  builder.addFieldOffset(3, descriptionOffset, 0);
  builder.addFieldOffset(4, buffersOffset, 0);
  return builder.endObject();
}

function readUint16(bytes, offset) {
  return new DataView(bytes.buffer, bytes.byteOffset + offset, 2).getUint16(0, true);
}

function readInt32(bytes, offset) {
  return new DataView(bytes.buffer, bytes.byteOffset + offset, 4).getInt32(0, true);
}

function fieldPos(bytes, tablePos, fieldIndex) {
  const vtableRelative = readInt32(bytes, tablePos);
  const vtablePos = tablePos - vtableRelative;
  const vtableLength = readUint16(bytes, vtablePos);
  const candidate = vtablePos + 4 + fieldIndex * 2;
  if (candidate + 2 > vtablePos + vtableLength) {
    return null;
  }
  const fieldOffset = readUint16(bytes, candidate);
  return fieldOffset === 0 ? null : tablePos + fieldOffset;
}

function readTableVectorEntry(bytes, tablePos, fieldIndex, index) {
  const vectorFieldPos = fieldPos(bytes, tablePos, fieldIndex);
  if (vectorFieldPos == null) {
    return null;
  }
  const vectorPos = vectorFieldPos + readInt32(bytes, vectorFieldPos);
  const length = readInt32(bytes, vectorPos);
  if (index < 0 || index >= length) {
    return null;
  }
  const entryPos = vectorPos + 4 + index * 4;
  return entryPos + readInt32(bytes, entryPos);
}

function writeBigUint64(bytes, offset, value) {
  new DataView(bytes.buffer, bytes.byteOffset + offset, 8).setBigUint64(0, BigInt(value), true);
}

function applyExternalBufferLocations(bytes, tensorBuffers) {
  const rootOffset = readInt32(bytes, 0);
  const firstDataOffset = bytes.byteLength;
  let cursor = firstDataOffset;
  for (let index = 0; index < tensorBuffers.length; index++) {
    const bufferPos = readTableVectorEntry(bytes, rootOffset, 4, index + 1);
    if (bufferPos == null) {
      throw new Error(`Missing buffer table entry ${index + 1} in external TFLite fixture.`);
    }
    const offsetFieldPos = fieldPos(bytes, bufferPos, 1);
    const sizeFieldPos = fieldPos(bytes, bufferPos, 2);
    if (offsetFieldPos == null || sizeFieldPos == null) {
      throw new Error(`External TFLite fixture buffer ${index + 1} is missing offset/size fields.`);
    }
    writeBigUint64(bytes, offsetFieldPos, cursor);
    writeBigUint64(bytes, sizeFieldPos, tensorBuffers[index].byteLength);
    cursor += tensorBuffers[index].byteLength;
  }
}

export function buildTfliteFixture(options = {}) {
  const tensors = Array.isArray(options.tensors) ? options.tensors : [];
  if (tensors.length === 0) {
    throw new Error('buildTfliteFixture requires at least one tensor.');
  }
  const externalBuffers = options.externalBuffers === true;

  const builder = new Builder(1024);
  const bufferOffsets = [createBuffer(builder, null, { external: false })];
  for (const tensor of tensors) {
    bufferOffsets.push(createBuffer(builder, tensor.data, { external: externalBuffers }));
  }

  const tensorOffsets = tensors.map((tensor, index) => createTensor(builder, {
    name: tensor.name,
    shape: tensor.shape,
    type: tensor.type,
    buffer: index + 1,
    omitTypeField: tensor.omitTypeField === true,
    isVariable: tensor.isVariable === true,
    quantization: tensor.quantization ?? null,
  }));
  const subgraphOffset = createSubGraph(builder, {
    tensorOffsets,
    name: options.subgraphName || 'main',
    inputs: Array.isArray(options.inputs) ? options.inputs : [],
    outputs: Array.isArray(options.outputs) ? options.outputs : [],
  });
  const modelOffset = createModel(builder, {
    version: Number.isFinite(options.version) ? Math.floor(options.version) : 3,
    subgraphOffset,
    bufferOffsets,
    description: options.description || 'fixture',
  });
  builder.finish(modelOffset, 'TFL3');
  const modelBytes = builder.asUint8Array().slice();
  if (!externalBuffers) {
    return modelBytes;
  }

  applyExternalBufferLocations(modelBytes, tensors.map((tensor) => tensor.data));
  const totalDataBytes = tensors.reduce((sum, tensor) => sum + tensor.data.byteLength, 0);
  const out = new Uint8Array(modelBytes.byteLength + totalDataBytes);
  out.set(modelBytes, 0);
  let cursor = modelBytes.byteLength;
  for (const tensor of tensors) {
    out.set(tensor.data, cursor);
    cursor += tensor.data.byteLength;
  }
  return out;
}
