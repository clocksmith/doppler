const TEXT_DECODER = new TextDecoder();

/** @type {import('./flatbuffer-tables.js').createFlatBufferTableReader} */
export function createFlatBufferTableReader({ readSlice, readInt32, readUint16 }) {
  /** @type {import('./flatbuffer-tables.js').FlatBufferTableReader['getTableFieldPos']} */
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

  /** @type {import('./flatbuffer-tables.js').FlatBufferTableReader['readStringAtField']} */
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

  /** @type {import('./flatbuffer-tables.js').FlatBufferTableReader['readTableVectorLength']} */
  async function readTableVectorLength(tablePos, fieldIndex) {
    const fieldPos = await getTableFieldPos(tablePos, fieldIndex);
    if (fieldPos == null) {
      return 0;
    }
    const vectorPos = fieldPos + await readInt32(fieldPos);
    return readInt32(vectorPos);
  }

  /** @type {import('./flatbuffer-tables.js').FlatBufferTableReader['readTableVectorEntry']} */
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
  return { getTableFieldPos, readStringAtField, readTableVectorLength, readTableVectorEntry };
}
