/**
 * SD3 joint text context assembly.
 *
 * @module experimental/diffusion/text-context
 */

import type { Tensor } from '../../gpu/tensor.js';

export interface Sd3TextContextCopy {
  /** Index into the source list: CLIP encoders in order, then T5. */
  source: number;
  sourceOffset: number;
  destOffset: number;
  byteLength: number;
}

export interface Sd3TextContextPlan {
  rows: number;
  jointDim: number;
  clipWidth: number;
  paddedPerClipRow: number;
  totalBytes: number;
  copies: Sd3TextContextCopy[];
}

export declare function bytesPerElementForDtype(dtype: string): number;

export declare function planSd3TextContext(input: {
  clipRows: number;
  clipDims: number[];
  t5Rows: number;
  jointDim: number;
  bytesPerElement: number;
}): Sd3TextContextPlan;

export declare function assembleSd3TextContext(input: {
  clipHidden: Tensor[];
  t5Hidden: Tensor;
}): Promise<Tensor>;
