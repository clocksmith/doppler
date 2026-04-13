import { Builder } from 'flatbuffers';

export const FIXTURE_LITERTLM_SECTION_TYPE = Object.freeze({
  NONE: 0,
  GenericBinaryData: 1,
  Deprecated: 2,
  TFLiteModel: 3,
  SP_Tokenizer: 4,
  LlmMetadataProto: 5,
  HF_Tokenizer_Zlib: 6,
  TFLiteWeights: 7,
});

const LITERT_BLOCK_BYTES = 16 * 1024;

function createOffsetVector(builder, offsets) {
  builder.startVector(4, offsets.length, 4);
  for (let index = offsets.length - 1; index >= 0; index--) {
    builder.addOffset(offsets[index]);
  }
  return builder.endVector();
}

function createSectionObject(builder, options) {
  builder.startObject(4);
  if (options.itemsOffset) {
    builder.addFieldOffset(0, options.itemsOffset, 0);
  }
  builder.addFieldInt64(1, BigInt(options.beginOffset), BigInt(0));
  builder.addFieldInt64(2, BigInt(options.endOffset), BigInt(0));
  builder.addFieldInt8(3, options.dataType, 0);
  return builder.endObject();
}

function createSectionMetadata(builder, sectionOffsets) {
  const objectsOffset = createOffsetVector(builder, sectionOffsets);
  builder.startObject(1);
  builder.addFieldOffset(0, objectsOffset, 0);
  return builder.endObject();
}

function createLiteRTLMMetaData(builder, sectionMetadataOffset) {
  builder.startObject(2);
  builder.addFieldOffset(1, sectionMetadataOffset, 0);
  return builder.endObject();
}

function alignUp(value, alignment) {
  const remainder = value % alignment;
  return remainder === 0 ? value : value + (alignment - remainder);
}

function concatBytes(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

export function buildLiteRTTaskFixture(files) {
  const normalizedFiles = Array.isArray(files) ? files : [];
  const localHeaders = [];
  const centralHeaders = [];
  let offset = 0;

  for (const file of normalizedFiles) {
    const nameBytes = new TextEncoder().encode(String(file.name || ''));
    const dataBytes = file.data instanceof Uint8Array ? file.data : new Uint8Array(file.data);
    const localHeader = new Uint8Array(30 + nameBytes.byteLength);
    const localView = new DataView(localHeader.buffer);
    localView.setUint32(0, 0x04034b50, true);
    localView.setUint16(4, 20, true);
    localView.setUint16(8, 0, true);
    localView.setUint16(10, 0, true);
    localView.setUint32(18, dataBytes.byteLength, true);
    localView.setUint32(22, dataBytes.byteLength, true);
    localView.setUint16(26, nameBytes.byteLength, true);
    localHeader.set(nameBytes, 30);
    localHeaders.push(localHeader, dataBytes);

    const centralHeader = new Uint8Array(46 + nameBytes.byteLength);
    const centralView = new DataView(centralHeader.buffer);
    centralView.setUint32(0, 0x02014b50, true);
    centralView.setUint16(4, 20, true);
    centralView.setUint16(6, 20, true);
    centralView.setUint16(10, 0, true);
    centralView.setUint32(20, dataBytes.byteLength, true);
    centralView.setUint32(24, dataBytes.byteLength, true);
    centralView.setUint16(28, nameBytes.byteLength, true);
    centralView.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);
    centralHeaders.push(centralHeader);

    offset += localHeader.byteLength + dataBytes.byteLength;
  }

  const centralDirectory = concatBytes(centralHeaders);
  const eocd = new Uint8Array(22);
  const eocdView = new DataView(eocd.buffer);
  eocdView.setUint32(0, 0x06054b50, true);
  eocdView.setUint16(8, normalizedFiles.length, true);
  eocdView.setUint16(10, normalizedFiles.length, true);
  eocdView.setUint32(12, centralDirectory.byteLength, true);
  eocdView.setUint32(16, offset, true);

  return concatBytes([...localHeaders, centralDirectory, eocd]);
}

export function buildLiteRTLmFixture(options = {}) {
  const sections = Array.isArray(options.sections) ? options.sections : [];
  const normalizedSections = [];
  let nextOffset = alignUp(32, LITERT_BLOCK_BYTES);
  for (const section of sections) {
    const dataBytes = section.data instanceof Uint8Array ? section.data : new Uint8Array(section.data);
    const beginOffset = alignUp(nextOffset, LITERT_BLOCK_BYTES);
    const endOffset = beginOffset + dataBytes.byteLength;
    normalizedSections.push({
      dataType: section.dataType,
      data: dataBytes,
      beginOffset,
      endOffset,
    });
    nextOffset = endOffset;
  }

  const builder = new Builder(1024);
  const sectionOffsets = normalizedSections.map((section) => createSectionObject(builder, {
    beginOffset: section.beginOffset,
    endOffset: section.endOffset,
    dataType: section.dataType,
    itemsOffset: 0,
  }));
  const sectionMetadataOffset = createSectionMetadata(builder, sectionOffsets);
  const metadataOffset = createLiteRTLMMetaData(builder, sectionMetadataOffset);
  builder.finish(metadataOffset);
  const headerBytes = builder.asUint8Array().slice();
  const headerEndOffset = 32 + headerBytes.byteLength;
  const totalSize = normalizedSections.length > 0
    ? normalizedSections[normalizedSections.length - 1].endOffset
    : headerEndOffset;
  const out = new Uint8Array(totalSize);
  out.set(new TextEncoder().encode('LITERTLM'), 0);
  const view = new DataView(out.buffer);
  view.setUint32(8, Number.isFinite(options.majorVersion) ? Math.floor(options.majorVersion) : 1, true);
  view.setUint32(12, Number.isFinite(options.minorVersion) ? Math.floor(options.minorVersion) : 0, true);
  view.setUint32(16, Number.isFinite(options.patchVersion) ? Math.floor(options.patchVersion) : 0, true);
  view.setBigUint64(24, BigInt(headerEndOffset), true);
  out.set(headerBytes, 32);
  for (const section of normalizedSections) {
    out.set(section.data, section.beginOffset);
  }
  return out;
}
