import type { BackwardKernelOptions } from './utils.js';

export interface GroupNormBackwardOptions extends BackwardKernelOptions {
  channels: number;
  height: number;
  width: number;
  numGroups: number;
  eps?: number;
}

export declare function runGroupNormBackward(
  input: any,
  weight: any,
  gradOutput: any,
  options: GroupNormBackwardOptions
): Promise<any>;

export declare function recordGroupNormBackward(
  recorder: any,
  input: any,
  weight: any,
  gradOutput: any,
  options: GroupNormBackwardOptions
): Promise<any>;
