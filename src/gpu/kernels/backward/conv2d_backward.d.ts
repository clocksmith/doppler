import type { BackwardKernelOptions } from './utils.js';

export interface Conv2DBackwardOptions extends BackwardKernelOptions {
  inChannels: number;
  outChannels: number;
  height: number;
  width: number;
  outHeight: number;
  outWidth: number;
  kernelH: number;
  kernelW: number;
  stride: number;
  pad: number;
  computeGradInput?: boolean;
  computeGradWeight?: boolean;
}

export declare function runConv2DBackward(
  input: any,
  weight: any,
  gradOutput: any,
  options: Conv2DBackwardOptions
): Promise<{ gradInput: any; gradWeight: any }>;

export declare function recordConv2DBackward(
  recorder: any,
  input: any,
  weight: any,
  gradOutput: any,
  options: Conv2DBackwardOptions
): Promise<{ gradInput: any; gradWeight: any }>;
