export interface W8A8StorageDescriptor {
  packing: 'w8a8-row-v1';
  blockShape: [1, number];
  blockBytes: number;
  paddedK: number;
  rowStrideBytes: number;
  scaleOffsetBytes: 0;
  dataOffsetBytes: 4;
}
export declare function createW8A8StorageDescriptor(columns: number): W8A8StorageDescriptor;
export declare function validateW8A8StorageDescriptor(storage: unknown, columns: number): W8A8StorageDescriptor;
export declare function validateW8A8StorageBytes(bytes: Uint8Array, shape: number[], storage: unknown): W8A8StorageDescriptor;
