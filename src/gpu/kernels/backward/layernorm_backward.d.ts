import type { BackwardKernelOptions } from './utils.js';

export interface LayerNormBackwardOptions extends BackwardKernelOptions {
  numTokens: number;
  hiddenSize: number;
  eps?: number;
}

export declare function runLayerNormBackward(
  input: any,
  weight: any,
  gradOutput: any,
  options: LayerNormBackwardOptions
): Promise<{ gradInput: any; gradWeight: any; gradBias: any }>;

export declare function recordLayerNormBackward(
  recorder: any,
  input: any,
  weight: any,
  gradOutput: any,
  options: LayerNormBackwardOptions
): Promise<{ gradInput: any; gradWeight: any; gradBias: any }>;
