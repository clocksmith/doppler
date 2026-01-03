import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';

export interface BackwardKernelOptions {
  count?: number;
  outputBuffer?: GPUBuffer | null;
}

export declare function runBackwardKernel(
  opName: string,
  input: Tensor,
  gradOutput: Tensor,
  uniformSize: number,
  writeUniforms: (view: DataView, count: number) => void,
  options?: BackwardKernelOptions
): Promise<Tensor>;

export declare function recordBackwardKernel(
  recorder: CommandRecorder,
  opName: string,
  input: Tensor,
  gradOutput: Tensor,
  uniformSize: number,
  writeUniforms: (view: DataView, count: number) => void,
  options?: BackwardKernelOptions
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
