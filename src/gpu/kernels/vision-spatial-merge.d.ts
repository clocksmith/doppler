import type { Tensor } from '../tensor.js';

export interface VisionSpatialMergeGeometry {
  gridHeight: number;
  gridWidth: number;
  hiddenSize: number;
  mergeSize: number;
}

export declare function runVisionSpatialMerge(
  input: Tensor,
  geometry: VisionSpatialMergeGeometry
): Promise<Tensor>;
