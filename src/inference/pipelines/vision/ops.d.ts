import type { Tensor } from '../../../gpu/tensor.js';

export interface VisionMatrixShape {
  M: number;
  K: number;
  N: number;
  bias?: Tensor | null;
}

export declare function doLayerNorm(
  input: GPUBuffer,
  weight: Tensor,
  bias: Tensor | null | undefined,
  opts: { seqLen: number; hiddenSize: number; eps: number }
): Promise<GPUBuffer>;

export declare function doMatmul(
  input: GPUBuffer,
  weight: Tensor,
  opts: VisionMatrixShape
): Promise<GPUBuffer>;

export declare function doGelu(
  input: GPUBuffer,
  opts: { count: number }
): Promise<GPUBuffer>;

export declare function doResidualAdd(
  left: GPUBuffer,
  right: GPUBuffer,
  opts: { count: number }
): Promise<GPUBuffer>;
