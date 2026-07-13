import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';

export interface GatedRmsNormBackwardOptions {
  rows: number;
  width: number;
  eps: number;
  gradInputBuffer?: GPUBuffer | null;
  gradGateBuffer?: GPUBuffer | null;
}

export interface GatedRmsNormBackwardResult {
  gradInput: Tensor;
  gradGate: Tensor;
}

export declare function runGatedRmsNormBackward(
  input: Tensor,
  gate: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: GatedRmsNormBackwardOptions
): Promise<GatedRmsNormBackwardResult>;

export declare function recordGatedRmsNormBackward(
  recorder: CommandRecorder,
  input: Tensor,
  gate: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: GatedRmsNormBackwardOptions
): Promise<GatedRmsNormBackwardResult>;
