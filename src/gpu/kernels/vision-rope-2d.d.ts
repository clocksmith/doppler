import type { Tensor } from '../tensor.js';

export interface VisionRope2DGeometry {
  numTokens: number;
  numHeads: number;
  headDim: number;
  gridHeight: number;
  gridWidth: number;
  ropeTheta: number;
  spatialMergeSize: number;
}

export declare function runVisionRope2D(
  input: Tensor,
  geometry: VisionRope2DGeometry
): Promise<Tensor>;
