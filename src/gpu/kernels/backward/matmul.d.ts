import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';

export interface MatmulBackwardDxOptions {
  alpha?: number;
  transposeB?: boolean;
  outputBuffer?: GPUBuffer | null;
}

export declare function resolveMatmulBackwardDxVariant(
  weight: { dtype?: string | null } | null
): 'default' | 'f16_weight' | 'q4k_weight';

export declare function runMatmulBackwardDx(
  dY: Tensor,
  W: Tensor,
  M: number,
  K: number,
  N: number,
  options?: MatmulBackwardDxOptions
): Promise<Tensor>;

export declare function recordMatmulBackwardDx(
  recorder: CommandRecorder,
  dY: Tensor,
  W: Tensor,
  M: number,
  K: number,
  N: number,
  options?: MatmulBackwardDxOptions
): Promise<Tensor>;

export declare function runMatmulTransposeA(
  A: Tensor,
  B: Tensor,
  M: number,
  N: number,
  K: number,
  options?: { alpha?: number; outputBuffer?: GPUBuffer | null }
): Promise<Tensor>;

export declare function recordMatmulTransposeA(
  recorder: CommandRecorder,
  A: Tensor,
  B: Tensor,
  M: number,
  N: number,
  K: number,
  options?: { alpha?: number; outputBuffer?: GPUBuffer | null }
): Promise<Tensor>;
