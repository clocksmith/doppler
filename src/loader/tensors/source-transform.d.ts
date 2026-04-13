import type { TensorLocation } from '../loader-types.js';

export declare function hasSourceTransform(location: TensorLocation | null | undefined): boolean;

export declare function getSourceTransformSpec(
  location: TensorLocation | null | undefined,
  tensorName: string
): ({
  kind: 'affine_dequant';
  elementCount: number;
  outputByteLength: number;
  sourceByteLength: number;
} | {
  kind: 'litert_rowwise_dequant';
  rows: number;
  cols: number;
  rawRowBytes: number;
  scaleRowBytes: number;
  targetRowBytes: number;
  outputByteLength: number;
  sourceByteLength: number;
  storageValuesPerByte: number;
} | {
  kind: 'litert_axis_dequant';
  rows: number;
  cols: number;
  storageRows: number;
  storageCols: number;
  rawStorageRowBytes: number;
  scaleRowBytes: number;
  targetRowBytes: number;
  outputByteLength: number;
  sourceByteLength: number;
  quantAxis: 0 | 1;
  storageValuesPerByte: number;
} | null);

export declare function materializeTensorSourceTransform(
  rawBytes: Uint8Array,
  location: TensorLocation,
  tensorName: string,
  options?: {
    scaleBytes?: Uint8Array | null;
    rowSumBytes?: Uint8Array | null;
    sumBytes?: Uint8Array | null;
    rowStart?: number | null;
    rowCount?: number | null;
    storageColumnStart?: number | null;
  }
): Uint8Array;
