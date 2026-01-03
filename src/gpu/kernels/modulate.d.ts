/**
 * @module gpu/kernels/modulate
 */

import type { Tensor } from '../tensor.js';

export interface ModulateOptions {
  numTokens: number;
  hiddenSize: number;
  scaleOffset?: number;
  shiftOffset?: number;
  gateOffset?: number;
  hasGate?: boolean;
  addOne?: boolean;
  outputBuffer?: GPUBuffer | null;
}

export declare function runModulate(
  input: Tensor,
  mod: Tensor,
  options: ModulateOptions
): Promise<Tensor>;

export declare function recordModulate(
  recorder: any,
  input: Tensor,
  mod: Tensor,
  options: ModulateOptions
): Promise<Tensor>;
