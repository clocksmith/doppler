/**
 * @module gpu/kernels/pixel_shuffle
 */

import type { Tensor } from '../tensor.js';

export interface PixelShuffleOptions {
  outChannels: number;
  outHeight: number;
  outWidth: number;
  gridWidth: number;
  gridHeight: number;
  patchSize: number;
  patchChannels?: number;
  outputBuffer?: GPUBuffer | null;
}

export declare function runPixelShuffle(
  input: Tensor,
  options: PixelShuffleOptions
): Promise<Tensor>;

export declare function recordPixelShuffle(
  recorder: any,
  input: Tensor,
  options: PixelShuffleOptions
): Promise<Tensor>;
