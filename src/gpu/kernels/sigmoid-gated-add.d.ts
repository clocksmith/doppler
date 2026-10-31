import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
export interface SigmoidGatedAddOptions {
  numTokens: number;
  hiddenSize: number;
  swigluLimit: number | null;
  outputBuffer?: GPUBuffer | null;
}
export declare function validateSigmoidGatedAdd(input: Tensor, gate: Tensor, residual: Tensor,
  options: SigmoidGatedAddOptions): void;
export declare function runSigmoidGatedAdd(input: Tensor, gate: Tensor, residual: Tensor,
  options: SigmoidGatedAddOptions): Promise<Tensor>;
export declare function recordSigmoidGatedAdd(recorder: CommandRecorder, input: Tensor, gate: Tensor,
  residual: Tensor, options: SigmoidGatedAddOptions): Promise<Tensor>;
