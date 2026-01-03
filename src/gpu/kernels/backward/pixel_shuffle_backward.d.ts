import type { BackwardKernelOptions } from './utils.js';

export interface PixelShuffleBackwardOptions extends BackwardKernelOptions {
  outChannels: number;
  outHeight: number;
  outWidth: number;
  gridWidth: number;
  gridHeight: number;
  patchSize: number;
  patchChannels: number;
}

export declare function runPixelShuffleBackward(
  gradOutput: any,
  options: PixelShuffleBackwardOptions
): Promise<any>;

export declare function recordPixelShuffleBackward(
  recorder: any,
  gradOutput: any,
  options: PixelShuffleBackwardOptions
): Promise<any>;
