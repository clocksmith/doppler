import type { Tensor } from '../tensor.js';

export interface VisionPatchGeometry {
  gridHeight: number;
  gridWidth: number;
  channels: number;
  patchSize: number;
  temporalPatchSize: number;
  hiddenSize: number;
}

export declare function runVisionPatchEmbed(
  imageData: Float32Array,
  weight: Tensor,
  bias: Tensor | null,
  geometry: VisionPatchGeometry
): Promise<Tensor>;
