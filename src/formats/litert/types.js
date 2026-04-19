const TEXT_DECODER = new TextDecoder();
const ZIP_TAIL_BYTES = 0xffff + 22;
const ZIP_EOCD_SIGNATURE = 0x06054b50;
const ZIP_CENTRAL_FILE_SIGNATURE = 0x02014b50;
const ZIP_LOCAL_FILE_SIGNATURE = 0x04034b50;
const LITERT_PAGE_BYTES = 16 * 1024;

export const LITERTLM_MAGIC = 'LITERTLM';
export const LITERT_TASK_DEFAULT_TFLITE_ENTRY = 'TF_LITE_PREFILL_DECODE';
export const LITERT_TASK_DEFAULT_TOKENIZER_MODEL_ENTRY = 'TOKENIZER_MODEL';
export const LITERT_TASK_DEFAULT_METADATA_ENTRY = 'METADATA';

const LITERTLM_SECTION_TYPE = Object.freeze({
  NONE: 0,
  GenericBinaryData: 1,
  Deprecated: 2,
  TFLiteModel: 3,
  SP_Tokenizer: 4,
  LlmMetadataProto: 5,
  HF_Tokenizer_Zlib: 6,
  TFLiteWeights: 7,
});

const LITERTLM_SECTION_TYPE_NAME = Object.freeze(Object.fromEntries(
  Object.entries(LITERTLM_SECTION_TYPE).map(([name, value]) => [value, name])
));

const LITERTLM_VALUE_TYPE = Object.freeze({
  NONE: 0,
  UInt8: 1,
  Int8: 2,
  UInt16: 3,
  Int16: 4,
  UInt32: 5,
  Int32: 6,
  Float32: 7,
  Bool: 8,
  StringValue: 9,
  UInt64: 10,
  Int64: 11,
  Double: 12,
});

function normalizeSourceLabel(source) {
  const raw = typeof source?.name === 'string' ? source.name.trim() : '';
  return raw || 'LiteRT source';
}

function decodeAscii(bytes) {
  return Array.from(bytes, (value) => String.fromCharCode(value)).join('');
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

function createPagedSourceReader(source) {
  if (!source || typeof source.readRange !== 'function') {
    throw new Error('LiteRT parser: source.readRange(offset, length) is required.');
  }
  const size = Number(source.size);
  if (!Number.isFinite(size) || size < 8) {
    throw new Error('LiteRT parser: source.size must be a finite byte length >= 8.');
  }
  const label = normalizeSourceLabel(source);
  const pageCache = new Map();

  async function loadPage(pageIndex) {
    const cached = pageCache.get(pageIndex);
    if (cached) {
      return cached;
    }
    const pageOffset = pageIndex * LITERT_PAGE_BYTES;
    const pageLength = Math.min(LITERT_PAGE_BYTES, size - pageOffset);
    const raw = await source.readRange(pageOffset, pageLength);
    const bytes = toUint8Array(raw, `${label} page ${pageIndex}`);
    if (bytes.byteLength !== pageLength) {
      throw new Error(
        `LiteRT parser: short read for ${label} page ${pageIndex}. ` +
        `Expected ${pageLength} bytes, got ${bytes.byteLength}.`
      );
    }
    pageCache.set(pageIndex, bytes);
    return bytes;
  }

  async function readSlice(offset, length) {
    if (!Number.isFinite(offset) || !Number.isFinite(length)) {
      throw new Error(`LiteRT parser: invalid read request (${offset}, ${length}).`);
    }
    const start = Math.max(0, Math.floor(offset));
    const byteLength = Math.max(0, Math.floor(length));
    if (start + byteLength > size) {
      throw new Error(
        `LiteRT parser: read exceeds ${label} bounds (offset=${start}, length=${byteLength}, size=${size}).`
      );
    }
    if (byteLength === 0) {
      return new Uint8Array(0);
    }
    const pageStart = Math.floor(start / LITERT_PAGE_BYTES);
    const pageEnd = Math.floor((start + byteLength - 1) / LITERT_PAGE_BYTES);
    if (pageStart === pageEnd) {
      const page = await loadPage(pageStart);
      const sliceOffset = start - pageStart * LITERT_PAGE_BYTES;
      return page.subarray(sliceOffset, sliceOffset + byteLength);
    }
    const out = new Uint8Array(byteLength);
    let writeOffset = 0;
    for (let pageIndex = pageStart; pageIndex <= pageEnd; pageIndex++) {
      const page = await loadPage(pageIndex);
      const pageBase = pageIndex * LITERT_PAGE_BYTES;
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

  async function readUint8(offset) {
    const bytes = await readSlice(offset, 1);
    return bytes[0];
  }

  async function readUint16(offset) {
    const bytes = await readSlice(offset, 2);
    return new DataView(bytes.buffer, bytes.byteOffset, 2).getUint16(0, true);
  }

  async function readInt16(offset) {
    const bytes = await readSlice(offset, 2);
    return new DataView(bytes.buffer, bytes.byteOffset, 2).getInt16(0, true);
  }

  async function readUint32(offset) {
    const bytes = await readSlice(offset, 4);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  }

  async function readInt32(offset) {
    const bytes = await readSlice(offset, 4);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getInt32(0, true);
  }

  async function readFloat32(offset) {
    const bytes = await readSlice(offset, 4);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getFloat32(0, true);
  }

  async function readBigUint64(offset) {
    const bytes = await readSlice(offset, 8);
    return new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true);
  }

  async function readBigInt64(offset) {
    const bytes = await readSlice(offset, 8);
    return new DataView(bytes.buffer, bytes.byteOffset, 8).getBigInt64(0, true);
  }

  async function readFloat64(offset) {
    const bytes = await readSlice(offset, 8);
    return new DataView(bytes.buffer, bytes.byteOffset, 8).getFloat64(0, true);
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

  async function readScalarAtField(tablePos, fieldIndex, type) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return null;
    }
    switch (type) {
      case 'uint8':
        return readUint8(fieldPos);
      case 'int8': {
        const raw = await readUint8(fieldPos);
        return raw > 127 ? raw - 256 : raw;
      }
      case 'uint16':
        return readUint16(fieldPos);
      case 'int16':
        return readInt16(fieldPos);
      case 'uint32':
        return readUint32(fieldPos);
      case 'int32':
        return readInt32(fieldPos);
      case 'uint64':
        return readBigUint64(fieldPos);
      case 'int64':
        return readBigInt64(fieldPos);
      case 'float32':
        return readFloat32(fieldPos);
      case 'float64':
        return readFloat64(fieldPos);
      case 'bool':
        return (await readUint8(fieldPos)) !== 0;
      default:
        throw new Error(`LiteRT parser: unsupported scalar type "${type}".`);
    }
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

  return {
    label,
    size,
    readSlice,
    readUint8,
    readUint16,
    readUint32,
    readInt32,
    readBigUint64,
    readBigInt64,
    getTableFieldPos,
    readStringAtField,
    readScalarAtField,
    readTableVectorLength,
    readTableVectorEntry,
  };
}

async function parseZipStoredEntries(reader) {
  const tailLength = Math.min(reader.size, ZIP_TAIL_BYTES);
  const tailStart = reader.size - tailLength;
  const tail = await reader.readSlice(tailStart, tailLength);

  let eocdOffset = -1;
  for (let index = tail.byteLength - 22; index >= 0; index--) {
    const signature = new DataView(tail.buffer, tail.byteOffset + index, 4).getUint32(0, true);
    if (signature === ZIP_EOCD_SIGNATURE) {
      eocdOffset = index;
      break;
    }
  }
  if (eocdOffset < 0) {
    throw new Error('LiteRT task parser: end-of-central-directory record not found.');
  }

  const eocdView = new DataView(tail.buffer, tail.byteOffset + eocdOffset, tail.byteLength - eocdOffset);
  const centralDirectorySize = eocdView.getUint32(12, true);
  const centralDirectoryOffset = eocdView.getUint32(16, true);
  const centralBytes = await reader.readSlice(centralDirectoryOffset, centralDirectorySize);
  const centralView = new DataView(centralBytes.buffer, centralBytes.byteOffset, centralBytes.byteLength);

  const entries = [];
  let offset = 0;
  while (offset < centralBytes.byteLength) {
    const signature = centralView.getUint32(offset, true);
    if (signature !== ZIP_CENTRAL_FILE_SIGNATURE) {
      throw new Error(
        `LiteRT task parser: invalid central directory signature at offset ${centralDirectoryOffset + offset}.`
      );
    }
    const compressionMethod = centralView.getUint16(offset + 10, true);
    const compressedSize = centralView.getUint32(offset + 20, true);
    const uncompressedSize = centralView.getUint32(offset + 24, true);
    const fileNameLength = centralView.getUint16(offset + 28, true);
    const extraLength = centralView.getUint16(offset + 30, true);
    const commentLength = centralView.getUint16(offset + 32, true);
    const localHeaderOffset = centralView.getUint32(offset + 42, true);
    const nameBytes = new Uint8Array(
      centralBytes.buffer,
      centralBytes.byteOffset + offset + 46,
      fileNameLength
    );
    const name = TEXT_DECODER.decode(nameBytes);
    if (compressionMethod !== 0) {
      throw new Error(
        `LiteRT task parser: zip entry "${name}" uses compression method ${compressionMethod}. ` +
        'Only stored (uncompressed) task archives are supported.'
      );
    }

    const localHeader = await reader.readSlice(localHeaderOffset, 30);
    const localView = new DataView(localHeader.buffer, localHeader.byteOffset, localHeader.byteLength);
    const localSignature = localView.getUint32(0, true);
    if (localSignature !== ZIP_LOCAL_FILE_SIGNATURE) {
      throw new Error(
        `LiteRT task parser: invalid local file header for "${name}" at offset ${localHeaderOffset}.`
      );
    }
    const localNameLength = localView.getUint16(26, true);
    const localExtraLength = localView.getUint16(28, true);
    const dataOffset = localHeaderOffset + 30 + localNameLength + localExtraLength;
    entries.push({
      name,
      compressionMethod,
      offset: dataOffset,
      size: uncompressedSize,
      compressedSize,
      localHeaderOffset,
    });

    offset += 46 + fileNameLength + extraLength + commentLength;
  }

  return entries;
}

async function parseLiteRTLMValue(reader, unionTypeId, tablePos) {
  switch (unionTypeId) {
    case LITERTLM_VALUE_TYPE.NONE:
      return null;
    case LITERTLM_VALUE_TYPE.UInt8:
      return reader.readScalarAtField(tablePos, 0, 'uint8');
    case LITERTLM_VALUE_TYPE.Int8:
      return reader.readScalarAtField(tablePos, 0, 'int8');
    case LITERTLM_VALUE_TYPE.UInt16:
      return reader.readScalarAtField(tablePos, 0, 'uint16');
    case LITERTLM_VALUE_TYPE.Int16:
      return reader.readScalarAtField(tablePos, 0, 'int16');
    case LITERTLM_VALUE_TYPE.UInt32:
      return reader.readScalarAtField(tablePos, 0, 'uint32');
    case LITERTLM_VALUE_TYPE.Int32:
      return reader.readScalarAtField(tablePos, 0, 'int32');
    case LITERTLM_VALUE_TYPE.Float32:
      return reader.readScalarAtField(tablePos, 0, 'float32');
    case LITERTLM_VALUE_TYPE.Bool:
      return reader.readScalarAtField(tablePos, 0, 'bool');
    case LITERTLM_VALUE_TYPE.StringValue:
      return reader.readStringAtField(tablePos, 0);
    case LITERTLM_VALUE_TYPE.UInt64: {
      const raw = await reader.readScalarAtField(tablePos, 0, 'uint64');
      const numeric = Number(raw);
      return Number.isSafeInteger(numeric) ? numeric : String(raw);
    }
    case LITERTLM_VALUE_TYPE.Int64: {
      const raw = await reader.readScalarAtField(tablePos, 0, 'int64');
      const numeric = Number(raw);
      return Number.isSafeInteger(numeric) ? numeric : String(raw);
    }
    case LITERTLM_VALUE_TYPE.Double:
      return reader.readScalarAtField(tablePos, 0, 'float64');
    default:
      return null;
  }
}

async function parseLiteRTLMKeyValuePair(reader, tablePos) {
  const key = await reader.readStringAtField(tablePos, 0);
  const valueType = await reader.readScalarAtField(tablePos, 1, 'uint8');
  const valueFieldPos = await reader.getTableFieldPos(tablePos, 2);
  const valueTablePos = valueFieldPos == null ? null : valueFieldPos + await reader.readInt32(valueFieldPos);
  return {
    key,
    valueType,
    value: valueTablePos == null ? null : await parseLiteRTLMValue(reader, valueType, valueTablePos),
  };
}

async function parseLiteRTLMSection(reader, tablePos) {
  const itemsLength = await reader.readTableVectorLength(tablePos, 0);
  const items = [];
  for (let index = 0; index < itemsLength; index++) {
    const itemPos = await reader.readTableVectorEntry(tablePos, 0, index);
    if (itemPos == null) {
      continue;
    }
    items.push(await parseLiteRTLMKeyValuePair(reader, itemPos));
  }
  const beginOffset = await reader.readScalarAtField(tablePos, 1, 'uint64');
  const endOffset = await reader.readScalarAtField(tablePos, 2, 'uint64');
  const dataType = await reader.readScalarAtField(tablePos, 3, 'uint8');
  return {
    beginOffset: Number(beginOffset),
    endOffset: Number(endOffset),
    size: Number(endOffset) - Number(beginOffset),
    dataType,
    dataTypeName: LITERTLM_SECTION_TYPE_NAME[dataType] ?? `UNKNOWN_${dataType}`,
    items,
  };
}

export async function parseLiteRTTaskFromSource(source) {
  const reader = createPagedSourceReader(source);
  const entries = await parseZipStoredEntries(reader);
  const entryMap = new Map(entries.map((entry) => [entry.name, entry]));
  return {
    entries,
    entryMap,
  };
}

export async function parseLiteRTLMFromSource(source) {
  const reader = createPagedSourceReader(source);
  const magic = decodeAscii(await reader.readSlice(0, 8));
  if (magic !== LITERTLM_MAGIC) {
    throw new Error(
      `LiteRT-LM parser: invalid magic "${magic || '(empty)'}". Expected "${LITERTLM_MAGIC}".`
    );
  }

  const majorVersion = await reader.readUint32(8);
  const minorVersion = await reader.readUint32(12);
  const patchVersion = await reader.readUint32(16);
  const headerEndOffset = Number(await reader.readBigUint64(24));
  if (!Number.isFinite(headerEndOffset) || headerEndOffset < 32 || headerEndOffset > reader.size) {
    throw new Error(`LiteRT-LM parser: invalid header end offset ${headerEndOffset}.`);
  }

  const headerBytes = await reader.readSlice(32, headerEndOffset - 32);
  const headerReader = createPagedSourceReader({
    name: `${normalizeSourceLabel(source)} header`,
    size: headerBytes.byteLength,
    async readRange(offset, length) {
      const start = Math.max(0, Math.floor(offset));
      const end = Math.min(headerBytes.byteLength, start + Math.max(0, Math.floor(length)));
      return headerBytes.slice(start, end);
    },
  });
  const rootOffset = await headerReader.readInt32(0);
  const metadataPos = rootOffset;
  const sectionMetadataFieldPos = await headerReader.getTableFieldPos(metadataPos, 1);
  if (sectionMetadataFieldPos == null) {
    throw new Error('LiteRT-LM parser: section metadata is required.');
  }
  const sectionMetadataPos = sectionMetadataFieldPos + await headerReader.readInt32(sectionMetadataFieldPos);
  const sectionCount = await headerReader.readTableVectorLength(sectionMetadataPos, 0);
  const sections = [];
  for (let index = 0; index < sectionCount; index++) {
    const sectionPos = await headerReader.readTableVectorEntry(sectionMetadataPos, 0, index);
    if (sectionPos == null) {
      continue;
    }
    sections.push(await parseLiteRTLMSection(headerReader, sectionPos));
  }

  return {
    majorVersion,
    minorVersion,
    patchVersion,
    headerEndOffset,
    sections,
  };
}

function normalizeUpper(value) {
  return String(value || '').trim().toUpperCase();
}

function findSectionItemValue(section, key) {
  const normalizedKey = normalizeUpper(key);
  for (const item of Array.isArray(section?.items) ? section.items : []) {
    if (normalizeUpper(item?.key) === normalizedKey) {
      return item?.value ?? null;
    }
  }
  return null;
}

export function findLiteRTLMSectionByType(parsed, dataTypeName) {
  const normalized = normalizeUpper(dataTypeName);
  return (Array.isArray(parsed?.sections) ? parsed.sections : []).filter((section) => (
    normalizeUpper(section?.dataTypeName) === normalized
  ));
}

export function findLiteRTLMTFLiteModelSection(parsed, modelType = 'TF_LITE_PREFILL_DECODE') {
  const sections = findLiteRTLMSectionByType(parsed, 'TFLiteModel');
  const normalizedModelType = normalizeUpper(modelType);
  const matched = sections.filter((section) => {
    const sectionModelType = normalizeUpper(findSectionItemValue(section, 'model_type'));
    return !sectionModelType || sectionModelType === normalizedModelType;
  });
  if (matched.length === 1) {
    return matched[0];
  }
  if (matched.length > 1) {
    throw new Error(
      `LiteRT-LM parser: multiple TFLiteModel sections matched model_type "${modelType}".`
    );
  }
  if (sections.length === 1) {
    return sections[0];
  }
  if (sections.length > 1) {
    throw new Error(
      `LiteRT-LM parser: multiple TFLiteModel sections found and none matched model_type "${modelType}".`
    );
  }
  return null;
}

export function findLiteRTLMTFLiteWeightsSection(parsed, modelType = 'TF_LITE_PREFILL_DECODE') {
  const sections = findLiteRTLMSectionByType(parsed, 'TFLiteWeights');
  const normalizedModelType = normalizeUpper(modelType);
  const matched = sections.filter((section) => {
    const sectionModelType = normalizeUpper(findSectionItemValue(section, 'model_type'));
    return !sectionModelType || sectionModelType === normalizedModelType;
  });
  if (matched.length === 1) {
    return matched[0];
  }
  if (matched.length > 1) {
    throw new Error(
      `LiteRT-LM parser: multiple TFLiteWeights sections matched model_type "${modelType}".`
    );
  }
  return null;
}

export function findLiteRTLMSentencePieceTokenizerSection(parsed) {
  const sections = findLiteRTLMSectionByType(parsed, 'SP_Tokenizer');
  if (sections.length > 1) {
    throw new Error('LiteRT-LM parser: multiple SP_Tokenizer sections are not supported.');
  }
  return sections[0] ?? null;
}

export function findLiteRTLMMetadataSection(parsed) {
  const sections = findLiteRTLMSectionByType(parsed, 'LlmMetadataProto');
  if (sections.length > 1) {
    throw new Error('LiteRT-LM parser: multiple LlmMetadataProto sections are not supported.');
  }
  return sections[0] ?? null;
}
