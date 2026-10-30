import type { WeightBuffer } from '../../gpu/weight-buffer.js';
export declare function loadQ40Packed(payload: ArrayBuffer | ArrayBufferView, location: {
  dtype: string; shape: number[]; layout?: string; size?: number;
  storage?: { packing: string; blockBytes: number; blockShape: number[] };
}, name: string, config: { representationRequirement?: { mode: string; dtypes: readonly string[] } }):
  { data: WeightBuffer; allocatedBuffers: GPUBuffer[] };
