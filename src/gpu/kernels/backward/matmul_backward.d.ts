import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './utils.js';

export interface MatmulBackwardOptions {
  M: number;
  N: number;
  K: number;
  transposeB?: boolean;
}

export interface MatmulBackwardResult {
  gradInput: Tensor;
  gradWeight: Tensor;
}

export declare function runMatmulBackward(
  input: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: MatmulBackwardOptions
): Promise<MatmulBackwardResult>;

export declare function recordMatmulBackward(): Promise<never>;
