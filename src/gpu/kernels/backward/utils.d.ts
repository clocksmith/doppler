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

/**
 * Factory that builds a backward-kernel runner pair from an op name
 * and a small spec (uniform-writer + workgroup sizing).
 */
export declare function createBackwardKernel(
  opName: string,
  spec: Record<string, unknown>
): {
  run: (...args: unknown[]) => Promise<Tensor>;
  record: (recorder: CommandRecorder, ...args: unknown[]) => Promise<Tensor>;
};

/**
 * Matmul-backward gradient w.r.t. input (dX = dY · Wᵀ). Submit + wait.
 */
export declare function runMatmulBackwardDx(
  dY: Tensor,
  W: Tensor,
  M: number,
  K: number,
  N: number,
  options?: { outputBuffer?: GPUBuffer | null }
): Promise<Tensor>;

/** Record runMatmulBackwardDx onto a CommandRecorder. */
export declare function recordMatmulBackwardDx(
  recorder: CommandRecorder,
  dY: Tensor,
  W: Tensor,
  M: number,
  K: number,
  N: number,
  options?: { outputBuffer?: GPUBuffer | null }
): Promise<Tensor>;
