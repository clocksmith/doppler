import type { Tensor } from '../tensor.js';

export interface VisionPositionEmbeddingGeometry {
  gridHeight: number;
  gridWidth: number;
  positionEmbeddingSize: number;
  hiddenSize: number;
}

export declare function runVisionPositionEmbedding(
  table: Tensor,
  geometry: VisionPositionEmbeddingGeometry
): Promise<Tensor>;
