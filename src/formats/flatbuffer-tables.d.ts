export interface FlatBufferReadPorts {
  readSlice(offset: number, length: number): Promise<Uint8Array>;
  readInt32(offset: number): Promise<number>;
  readUint16(offset: number): Promise<number>;
}

export interface FlatBufferTableReader {
  getTableFieldPos(tablePos: number, fieldIndex: number): Promise<number | null>;
  readStringAtField(tablePos: number, fieldIndex: number): Promise<string | null>;
  readTableVectorLength(tablePos: number, fieldIndex: number): Promise<number>;
  readTableVectorEntry(tablePos: number, fieldIndex: number, index: number): Promise<number | null>;
}

export declare function createFlatBufferTableReader(ports: FlatBufferReadPorts): FlatBufferTableReader;
