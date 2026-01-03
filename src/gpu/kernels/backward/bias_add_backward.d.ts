import type { BackwardKernelOptions } from './utils.js';

export interface BiasAddBackwardOptions extends BackwardKernelOptions {
  numTokens: number;
  dim: number;
}

export declare function runBiasAddBackward(
  gradOutput: any,
  options: BiasAddBackwardOptions
): Promise<any>;

export declare function recordBiasAddBackward(
  recorder: any,
  gradOutput: any,
  options: BiasAddBackwardOptions
): Promise<any>;
