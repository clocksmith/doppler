import type { W8A8Policy } from '../config/schema/w8a8.schema.js';
import type { W8A8StorageDescriptor } from '../formats/rdrr/w8a8-storage.js';
export declare function roundW8A8Code(value: number, scale: number): number;
export declare function quantizeToW8A8Rows(values: Float32Array, shape: number[], policy: W8A8Policy): {
  quantized: Uint8Array;
  storage: W8A8StorageDescriptor;
};
