import type { LayerWeights } from '../types.js';

export declare function hasQ4KMaterialization(weight: unknown): boolean;

export declare function canUseRmsNormWideTileProjectionFusion(
  layerWeights: LayerWeights | null | undefined,
  reusesSharedKV: boolean
): boolean;
