import type { Tensor } from '../tensor.js';

export interface VisionSpatialMergeGeometry {
  gridHeight: number;
  gridWidth: number;
  hiddenSize: number;
  mergeSize: number;
  channelFirst?: boolean;
  inputBlockMajor?: boolean;
}

export declare function planVisionSpatialMergeDispatch(
  outputElements: number
): [number, number, 1];

export declare function runVisionSpatialMerge(
  input: Tensor,
  geometry: VisionSpatialMergeGeometry
): Promise<Tensor>;
