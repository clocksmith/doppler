import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './utils.js';

export interface MatmulBackwardOptions {
  M: number;
  N: number;
  K: number;
  transposeB?: boolean;
  computeGradInput?: boolean;
  computeGradWeight?: boolean;
}

export interface MatmulBackwardResult {
  gradInput: Tensor | null;
  gradWeight: Tensor | null;
}

export declare function runMatmulBackward(
  input: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: MatmulBackwardOptions
): Promise<MatmulBackwardResult>;

export declare function recordMatmulBackward(
  recorder: CommandRecorder,
  input: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: MatmulBackwardOptions
): Promise<MatmulBackwardResult>;
