import type { Tensor } from '../tensor.js';

export interface VisionAveragePoolGeometry {
  gridHeight: number;
  gridWidth: number;
  hiddenSize: number;
  poolingSize: number;
}

export declare function runVisionAveragePool(
  input: Tensor,
  geometry: VisionAveragePoolGeometry
): Promise<Tensor>;
