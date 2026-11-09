import type { WeightBuffer, WeightLayout } from '../gpu/weight-buffer.js';

export declare function isQ4KLocationDtype(dtype: string): boolean;
export declare function toUint8Array(data: unknown): Uint8Array | null;
export declare function acquireUploadedBuffer(
  data: ArrayBufferView,
  label: string,
  owner?: { add?: (buffer: GPUBuffer) => unknown } | null
): GPUBuffer;
export declare function loadReferenceQ4KWeight(
  ctx: { loadTensor: (name: string, cache: boolean, raw: boolean) => Promise<unknown>; gpuBuffers?: { add?: (buffer: GPUBuffer) => unknown } },
  request: {
    name: string;
    location: { dtype?: unknown; shape?: unknown; size: number } | null | undefined;
    layout: WeightLayout | (() => WeightLayout);
    label?: string;
  }
): Promise<WeightBuffer | null>;
