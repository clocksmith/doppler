export interface ByteRangeSource {
  name?: string;
  size: number;
  readRange(offset: number, length: number): Promise<ArrayBuffer | Uint8Array> | ArrayBuffer | Uint8Array;
}

export interface PagedByteReader {
  readonly label: string;
  readonly size: number;
  readonly pageBytes: number;
  readSlice(offset: number, length: number): Promise<Uint8Array>;
  readUint8(offset: number): Promise<number>;
  readInt16(offset: number): Promise<number>;
  readUint16(offset: number): Promise<number>;
  readInt32(offset: number): Promise<number>;
  readUint32(offset: number): Promise<number>;
  readFloat32(offset: number): Promise<number>;
  readFloat64(offset: number): Promise<number>;
  readBigInt64(offset: number): Promise<bigint>;
  readBigUint64(offset: number): Promise<bigint>;
}

export declare function describeSource(source: { name?: unknown } | null | undefined, defaultLabel: string): string;

export declare function createPagedByteReader(
  source: ByteRangeSource,
  options: { pageBytes: number; parser: string; defaultLabel: string }
): PagedByteReader;
