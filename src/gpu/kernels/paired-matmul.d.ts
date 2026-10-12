import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { MatmulOptions } from './matmul.js';
/** Paired reduction with unchanged materialized weight values. */
export declare function executePairedMatmul(recorder: CommandRecorder | null, input: Tensor,
  weight: GPUBuffer | WeightBuffer, M: number, N: number, K: number,
  options: MatmulOptions & { outputPrecision?: 'f32' | 'two-f32' }): Promise<Tensor>;
