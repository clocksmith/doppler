import type { BackwardKernelOptions } from './utils.js';

export interface Upsample2DBackwardOptions extends BackwardKernelOptions {
  channels: number;
  inHeight: number;
  inWidth: number;
  outHeight: number;
  outWidth: number;
  scale: number;
}

export declare function runUpsample2DBackward(
  gradOutput: any,
  options: Upsample2DBackwardOptions
): Promise<any>;

export declare function recordUpsample2DBackward(
  recorder: any,
  gradOutput: any,
  options: Upsample2DBackwardOptions
): Promise<any>;
