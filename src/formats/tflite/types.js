const TEXT_DECODER = new TextDecoder();
const TFLITE_PAGE_BYTES = 16 * 1024;

export const TFLITE_FILE_IDENTIFIER = 'TFL3';

export const TFLITE_TENSOR_TYPE = Object.freeze({
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

export const TFLITE_TENSOR_TYPE_NAME = Object.freeze(Object.fromEntries(
  Object.entries(TFLITE_TENSOR_TYPE).map(([name, value]) => [value, name])
));

export const TFLITE_TENSOR_DTYPE_MAP = Object.freeze({
  [TFLITE_TENSOR_TYPE.FLOAT32]: 'F32',
  [TFLITE_TENSOR_TYPE.FLOAT16]: 'F16',
  [TFLITE_TENSOR_TYPE.BFLOAT16]: 'BF16',
});

export const TFLITE_TENSOR_SOURCE_DTYPE_MAP = Object.freeze({
  [TFLITE_TENSOR_TYPE.FLOAT32]: 'F32',
  [TFLITE_TENSOR_TYPE.FLOAT16]: 'F16',
  [TFLITE_TENSOR_TYPE.BFLOAT16]: 'BF16',
  [TFLITE_TENSOR_TYPE.INT8]: 'INT8',
  [TFLITE_TENSOR_TYPE.UINT8]: 'UINT8',
  [TFLITE_TENSOR_TYPE.INT4]: 'INT4',
});

export const TFLITE_DTYPE_SIZE = Object.freeze({
  F32: 4,
  F16: 2,
  BF16: 2,
});

function normalizeParseOptions(options) {
  return {
    allowPackedQuantization: options?.allowPackedQuantization === true,
  };
}

function toUint8Array(value, label) {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  throw new Error(`${label} must return ArrayBuffer or Uint8Array.`);
}

function normalizeSourceLabel(source) {
  const raw = typeof source?.name === 'string' ? source.name.trim() : '';
  return raw || 'TFLite source';
}

function decodeAscii(bytes) {
  return Array.from(bytes, (value) => String.fromCharCode(value)).join('');
}

function computeTensorElementCount(shape, tensorName) {
  if (!Array.isArray(shape)) {
    throw new Error(`TFLite tensor "${tensorName}" is missing shape.`);
  }
  let total = 1;
  for (let index = 0; index < shape.length; index++) {
    const rawDim = Number(shape[index]);
    if (!Number.isFinite(rawDim) || Math.floor(rawDim) !== rawDim || rawDim < 0) {
      throw new Error(
        `TFLite tensor "${tensorName}" has invalid shape[${index}] (${shape[index]}).`
      );
    }
    total *= rawDim;
  }
  return total;
}

function computeTFLiteSourceByteSize(elementCount, sourceDtype, tensorName) {
  if (!Number.isInteger(elementCount) || elementCount < 0) {
    throw new Error(`TFLite tensor "${tensorName}" has invalid element count (${elementCount}).`);
  }
  if (sourceDtype === 'INT4') {
    return Math.ceil(elementCount / 2);
  }
  const byteSize = TFLITE_DTYPE_SIZE[sourceDtype];
  if (Number.isInteger(byteSize) && byteSize >= 0) {
    return elementCount * byteSize;
  }
  if (sourceDtype === 'INT8' || sourceDtype === 'UINT8') {
    return elementCount;
  }
  throw new Error(
    `TFLite tensor "${tensorName}" uses unsupported source dtype ${sourceDtype}.`
  );
}

function resolveDirectSourceQuantization(tensors) {
  const dtypes = new Set();
  for (const tensor of tensors) {
    const dtype = String(tensor?.dtype || '').trim().toUpperCase();
    if (dtype) {
      dtypes.add(dtype);
    }
  }
  if (dtypes.size === 0) {
    return null;
  }
  if (dtypes.size === 1) {
    return [...dtypes][0];
  }
  const allSupportedFloat = Array.from(dtypes).every((dtype) => dtype === 'F16' || dtype === 'BF16' || dtype === 'F32');
  return allSupportedFloat ? 'F32' : null;
}

function createMemoryRangeSource(buffer, label = 'TFLite buffer') {
  const bytes = toUint8Array(buffer, label);
  return {
    name: label,
    size: bytes.byteLength,
    async readRange(offset, length) {
      if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
        return new Uint8Array(0);
      }
      const start = Math.max(0, Math.floor(offset));
      const end = Math.min(bytes.byteLength, start + Math.max(0, Math.floor(length)));
      return bytes.slice(start, end);
    },
  };
}

function createPagedSourceReader(source) {
  if (!source || typeof source.readRange !== 'function') {
    throw new Error('TFLite parser: source.readRange(offset, length) is required.');
  }
  const size = Number(source.size);
  if (!Number.isFinite(size) || size < 8) {
    throw new Error('TFLite parser: source.size must be a finite byte length >= 8.');
  }
  const label = normalizeSourceLabel(source);
  const pageCache = new Map();

  async function loadPage(pageIndex) {
    const cached = pageCache.get(pageIndex);
    if (cached) {
      return cached;
    }
    const pageOffset = pageIndex * TFLITE_PAGE_BYTES;
    const pageLength = Math.min(TFLITE_PAGE_BYTES, size - pageOffset);
    const raw = await source.readRange(pageOffset, pageLength);
    const bytes = toUint8Array(raw, `${label} page ${pageIndex}`);
    if (bytes.byteLength !== pageLength) {
      throw new Error(
        `TFLite parser: short read for ${label} page ${pageIndex}. ` +
        `Expected ${pageLength} bytes, got ${bytes.byteLength}.`
      );
    }
    pageCache.set(pageIndex, bytes);
    return bytes;
  }

  async function readSlice(offset, length) {
    if (!Number.isFinite(offset) || !Number.isFinite(length)) {
      throw new Error(`TFLite parser: invalid read request (${offset}, ${length}).`);
    }
    const start = Math.max(0, Math.floor(offset));
    const byteLength = Math.max(0, Math.floor(length));
    if (start + byteLength > size) {
      throw new Error(
        `TFLite parser: read exceeds ${label} bounds (offset=${start}, length=${byteLength}, size=${size}).`
      );
    }
    if (byteLength === 0) {
      return new Uint8Array(0);
    }
    const pageStart = Math.floor(start / TFLITE_PAGE_BYTES);
    const pageEnd = Math.floor((start + byteLength - 1) / TFLITE_PAGE_BYTES);
    if (pageStart === pageEnd) {
      const page = await loadPage(pageStart);
      const sliceOffset = start - pageStart * TFLITE_PAGE_BYTES;
      return page.subarray(sliceOffset, sliceOffset + byteLength);
    }
    const out = new Uint8Array(byteLength);
    let writeOffset = 0;
    for (let pageIndex = pageStart; pageIndex <= pageEnd; pageIndex++) {
      const page = await loadPage(pageIndex);
      const pageBase = pageIndex * TFLITE_PAGE_BYTES;
      const sliceStart = pageIndex === pageStart ? start - pageBase : 0;
      const sliceEnd = pageIndex === pageEnd
        ? (start + byteLength) - pageBase
        : page.byteLength;
      const slice = page.subarray(sliceStart, sliceEnd);
      out.set(slice, writeOffset);
      writeOffset += slice.byteLength;
    }
    return out;
  }

  async function readInt32(offset) {
    const bytes = await readSlice(offset, 4);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getInt32(0, true);
  }

  async function readUint32(offset) {
    const bytes = await readSlice(offset, 4);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  }

  async function readUint16(offset) {
    const bytes = await readSlice(offset, 2);
    return new DataView(bytes.buffer, bytes.byteOffset, 2).getUint16(0, true);
  }

  async function readUint8(offset) {
    const bytes = await readSlice(offset, 1);
    return bytes[0];
  }

  async function readFloat32(offset) {
    const bytes = await readSlice(offset, 4);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getFloat32(0, true);
  }

  async function readBigInt64(offset) {
    const bytes = await readSlice(offset, 8);
    return new DataView(bytes.buffer, bytes.byteOffset, 8).getBigInt64(0, true);
  }

  async function readBigUint64(offset) {
    const bytes = await readSlice(offset, 8);
    return new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true);
  }

  async function getTableFieldPos(tablePos, fieldIndex) {
    const vtableRelative = await readInt32(tablePos);
    const vtablePos = tablePos - vtableRelative;
    const vtableLength = await readUint16(vtablePos);
    const fieldPos = vtablePos + 4 + fieldIndex * 2;
    if (fieldPos + 2 > vtablePos + vtableLength) {
      return null;
    }
    const fieldOffset = await readUint16(fieldPos);
    return fieldOffset === 0 ? null : tablePos + fieldOffset;
  }

  async function readStringAtField(tablePos, fieldIndex) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return null;
    }
    const stringPos = fieldPos + await readInt32(fieldPos);
    const stringLength = await readInt32(stringPos);
    const bytes = await readSlice(stringPos + 4, stringLength);
    return TEXT_DECODER.decode(bytes);
  }

  async function readInt32VectorAtField(tablePos, fieldIndex) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return [];
    }
    const vectorPos = fieldPos + await readInt32(fieldPos);
    const length = await readInt32(vectorPos);
    const values = [];
    for (let index = 0; index < length; index++) {
      values.push(await readInt32(vectorPos + 4 + index * 4));
    }
    return values;
  }

  async function readFloat32VectorAtField(tablePos, fieldIndex) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return [];
    }
    const vectorPos = fieldPos + await readInt32(fieldPos);
    const length = await readInt32(vectorPos);
    const values = [];
    for (let index = 0; index < length; index++) {
      values.push(await readFloat32(vectorPos + 4 + index * 4));
    }
    return values;
  }

  async function readInt64VectorAtField(tablePos, fieldIndex) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return [];
    }
    const vectorPos = fieldPos + await readInt32(fieldPos);
    const length = await readInt32(vectorPos);
    const values = [];
    for (let index = 0; index < length; index++) {
      const value = await readBigInt64(vectorPos + 4 + index * 8);
      const numeric = Number(value);
      if (!Number.isSafeInteger(numeric)) {
        throw new Error(
          `TFLite parser: int64 quantization value ${String(value)} exceeds JS safe integer range.`
        );
      }
      values.push(numeric);
    }
    return values;
  }

  async function readTableVectorLength(tablePos, fieldIndex) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return 0;
    }
    const vectorPos = fieldPos + await readInt32(fieldPos);
    return readInt32(vectorPos);
  }

  async function readTableVectorEntry(tablePos, fieldIndex, index) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return null;
    }
    const vectorPos = fieldPos + await readInt32(fieldPos);
    const length = await readInt32(vectorPos);
    if (index < 0 || index >= length) {
      return null;
    }
    const entryPos = vectorPos + 4 + index * 4;
    return entryPos + await readInt32(entryPos);
  }

  async function readByteVectorLocationAtField(tablePos, fieldIndex) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return null;
    }
    const vectorPos = fieldPos + await readInt32(fieldPos);
    return {
      offset: vectorPos + 4,
      size: await readInt32(vectorPos),
    };
  }

  return {
    label,
    size,
    readSlice,
    readInt32,
    readUint32,
    readUint16,
    readUint8,
    readFloat32,
    readBigInt64,
    readBigUint64,
    getTableFieldPos,
    readStringAtField,
    readInt32VectorAtField,
    readFloat32VectorAtField,
    readInt64VectorAtField,
    readTableVectorLength,
    readTableVectorEntry,
    readByteVectorLocationAtField,
  };
}

async function readTensorQuantization(reader, tensorPos, tensorName, sourceDtype) {
  const quantizationPos = await reader.getTableFieldPos(tensorPos, 4);
  if (quantizationPos == null) {
    throw new Error(
      `TFLite parser: quantized tensor "${tensorName}" is missing quantization parameters. ` +
      `Packed LiteRT-LM companion tensors such as "${tensorName}.sum_i" and ` +
      `"${tensorName}_quantized_scale" are not supported in direct-source mode yet.`
    );
  }
  const tablePos = quantizationPos + await reader.readInt32(quantizationPos);
  const scales = await reader.readFloat32VectorAtField(tablePos, 2);
  const zeroPoints = await reader.readInt64VectorAtField(tablePos, 3);
  const quantizedDimensionFieldPos = await reader.getTableFieldPos(tablePos, 5);
  const quantizedDimension = quantizedDimensionFieldPos == null
    ? 0
    : await reader.readInt32(quantizedDimensionFieldPos);

  if (scales.length !== 1) {
    throw new Error(
      `TFLite parser: quantized tensor "${tensorName}" must use exactly one scale value. ` +
      `Found ${scales.length}. Per-channel quantization is not supported in direct-source mode.`
    );
  }
  if (zeroPoints.length > 1) {
    throw new Error(
      `TFLite parser: quantized tensor "${tensorName}" must use at most one zero_point value. ` +
      `Found ${zeroPoints.length}. Per-channel quantization is not supported in direct-source mode.`
    );
  }
  if (quantizedDimension !== 0) {
    throw new Error(
      `TFLite parser: quantized tensor "${tensorName}" uses quantized_dimension=${quantizedDimension}. ` +
      'Only per-tensor affine quantization is supported in direct-source mode.'
    );
  }

  const scale = Number(scales[0]);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(
      `TFLite parser: quantized tensor "${tensorName}" has invalid scale ${scales[0]}.`
    );
  }
  const zeroPoint = zeroPoints.length === 0 ? 0 : Number(zeroPoints[0]);
  if (!Number.isSafeInteger(zeroPoint)) {
    throw new Error(
      `TFLite parser: quantized tensor "${tensorName}" has invalid zero_point ${zeroPoints[0]}.`
    );
  }

  return {
    kind: 'affine_dequant',
    scheme: 'per_tensor_affine',
    sourceDtype,
    targetDtype: 'F16',
    scale,
    zeroPoint,
  };
}

async function readBufferLocation(reader, bufferPos, tensorName) {
  const inlineLocation = await reader.readByteVectorLocationAtField(bufferPos, 0);
  if (inlineLocation && inlineLocation.size > 0) {
    return inlineLocation;
  }

  const offsetFieldPos = await reader.getTableFieldPos(bufferPos, 1);
  const sizeFieldPos = await reader.getTableFieldPos(bufferPos, 2);
  if (offsetFieldPos == null || sizeFieldPos == null) {
    return null;
  }

  const offsetRaw = await reader.readBigUint64(offsetFieldPos);
  const sizeRaw = await reader.readBigUint64(sizeFieldPos);
  const offset = Number(offsetRaw);
  const size = Number(sizeRaw);
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new Error(
      `TFLite parser: tensor "${tensorName}" uses buffer offset ${String(offsetRaw)} outside JS safe integer range.`
    );
  }
  if (!Number.isSafeInteger(size) || size < 0) {
    throw new Error(
      `TFLite parser: tensor "${tensorName}" uses buffer size ${String(sizeRaw)} outside JS safe integer range.`
    );
  }
  if (offset + size > reader.size) {
    throw new Error(
      `TFLite parser: tensor "${tensorName}" buffer range exceeds file bounds (offset=${offset}, size=${size}, fileSize=${reader.size}).`
    );
  }
  return size > 0 ? { offset, size } : null;
}

async function parseTFLiteReader(reader, options = {}) {
  const parseOptions = normalizeParseOptions(options);
  const rootOffset = await reader.readInt32(0);
  const fileIdentifier = decodeAscii(await reader.readSlice(4, 4));
  if (fileIdentifier !== TFLITE_FILE_IDENTIFIER) {
    throw new Error(
      `TFLite parser: invalid file identifier "${fileIdentifier || '(empty)'}". ` +
      `Expected "${TFLITE_FILE_IDENTIFIER}".`
    );
  }

  const modelPos = rootOffset;
  const versionFieldPos = await reader.getTableFieldPos(modelPos, 0);
  const schemaVersion = versionFieldPos == null ? 0 : await reader.readUint32(versionFieldPos);
  const description = await reader.readStringAtField(modelPos, 3);
  const subgraphCount = await reader.readTableVectorLength(modelPos, 2);
  const bufferCount = await reader.readTableVectorLength(modelPos, 4);

  if (subgraphCount < 1) {
    throw new Error('TFLite parser: model must contain at least one subgraph.');
  }
  if (bufferCount < 1) {
    throw new Error('TFLite parser: model must contain at least one buffer.');
  }

  const mainSubgraphPos = await reader.readTableVectorEntry(modelPos, 2, 0);
  if (mainSubgraphPos == null) {
    throw new Error('TFLite parser: failed to resolve the main subgraph.');
  }
  const mainSubgraphName = await reader.readStringAtField(mainSubgraphPos, 4);
  const tensorCount = await reader.readTableVectorLength(mainSubgraphPos, 0);
  const tensors = [];
  const seenNames = new Set();

  for (let index = 0; index < tensorCount; index++) {
    const tensorPos = await reader.readTableVectorEntry(mainSubgraphPos, 0, index);
    if (tensorPos == null) {
      continue;
    }
    const name = await reader.readStringAtField(tensorPos, 3);
    if (!name) {
      continue;
    }
    if (seenNames.has(name)) {
      throw new Error(`TFLite parser: duplicate tensor name "${name}" in the main subgraph.`);
    }
    seenNames.add(name);

    const bufferFieldPos = await reader.getTableFieldPos(tensorPos, 2);
    const bufferIndex = bufferFieldPos == null ? 0 : await reader.readUint32(bufferFieldPos);
    if (bufferIndex === 0) {
      continue;
    }
    if (bufferIndex >= bufferCount) {
      throw new Error(
        `TFLite parser: tensor "${name}" references buffer ${bufferIndex}, ` +
        `but the model only has ${bufferCount} buffers.`
      );
    }

    const bufferPos = await reader.readTableVectorEntry(modelPos, 4, bufferIndex);
    if (bufferPos == null) {
      throw new Error(`TFLite parser: missing buffer table entry ${bufferIndex} for tensor "${name}".`);
    }
    const dataLocation = await readBufferLocation(reader, bufferPos, name);
    if (!dataLocation || dataLocation.size <= 0) {
      continue;
    }

    const typeFieldPos = await reader.getTableFieldPos(tensorPos, 1);
    const dtypeId = typeFieldPos == null ? TFLITE_TENSOR_TYPE.FLOAT32 : await reader.readUint8(typeFieldPos);
    const dtypeName = TFLITE_TENSOR_TYPE_NAME[dtypeId] ?? `UNKNOWN_${dtypeId}`;
    const mappedSourceDtype = TFLITE_TENSOR_SOURCE_DTYPE_MAP[dtypeId] ?? null;
    const sourceDtype = mappedSourceDtype ?? (parseOptions.allowPackedQuantization ? dtypeName : null);
    if (!sourceDtype) {
      throw new Error(`TFLite parser: tensor "${name}" uses unsupported constant dtype ${dtypeName}.`);
    }

    const isVariableFieldPos = await reader.getTableFieldPos(tensorPos, 5);
    const isVariable = isVariableFieldPos == null ? false : await reader.readUint8(isVariableFieldPos) !== 0;
    if (isVariable) {
      throw new Error(
        `TFLite parser: variable tensor "${name}" is not supported in direct-source mode.`
      );
    }

    const shape = await reader.readInt32VectorAtField(tensorPos, 0);
    const elementCount = computeTensorElementCount(shape, name);
    const isQuantizedSourceDtype = (
      mappedSourceDtype === 'INT8'
      || mappedSourceDtype === 'UINT8'
      || mappedSourceDtype === 'INT4'
    );
    let sourceTransform = null;
    if (isQuantizedSourceDtype) {
      try {
        sourceTransform = await readTensorQuantization(reader, tensorPos, name, mappedSourceDtype);
      } catch (error) {
        if (!parseOptions.allowPackedQuantization) {
          throw error;
        }
      }
    }
    const dtype = sourceTransform?.targetDtype
      ?? (TFLITE_TENSOR_DTYPE_MAP[dtypeId] ?? (parseOptions.allowPackedQuantization ? sourceDtype : null));
    if (!dtype && !parseOptions.allowPackedQuantization) {
      throw new Error(
        `TFLite parser: tensor "${name}" uses unsupported constant dtype ${dtypeName}.`
      );
    }
    const expectedBytes = mappedSourceDtype
      ? computeTFLiteSourceByteSize(elementCount, mappedSourceDtype, name)
      : null;
    if (expectedBytes !== null && expectedBytes !== dataLocation.size && !parseOptions.allowPackedQuantization) {
      throw new Error(
        `TFLite parser: tensor "${name}" byte size mismatch. ` +
        `Expected ${expectedBytes} bytes from shape ${JSON.stringify(shape)} and dtype ${sourceDtype}, ` +
        `found ${dataLocation.size}.`
      );
    }

    tensors.push({
      name,
      shape,
      dtype: dtype ?? sourceDtype,
      dtypeId,
      sourceDtype,
      offset: dataLocation.offset,
      size: dataLocation.size,
      buffer: bufferIndex,
      subgraphIndex: 0,
      isVariable: false,
      ...(sourceTransform ? { sourceTransform } : {}),
    });
  }

  if (tensors.length === 0) {
    throw new Error(
      'TFLite parser: no supported constant tensors were found in the main subgraph.'
    );
  }

  return {
    schemaVersion,
    description: description || null,
    subgraphCount,
    mainSubgraphName: mainSubgraphName || null,
    tensors,
    sourceQuantization: resolveDirectSourceQuantization(tensors),
  };
}

export async function parseTFLite(buffer, options = {}) {
  return parseTFLiteReader(
    createPagedSourceReader(createMemoryRangeSource(buffer)),
    options
  );
}

export async function parseTFLiteFromSource(source, options = {}) {
  return parseTFLiteReader(createPagedSourceReader(source), options);
}
