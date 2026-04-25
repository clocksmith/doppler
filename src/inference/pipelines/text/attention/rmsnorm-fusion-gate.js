import { isWeightBuffer } from '../../../../gpu/weight-buffer.js';

export function hasQ4KMaterialization(weight) {
  return isWeightBuffer(weight) && Boolean(weight.materializations?.q4k?.buffer);
}

export function canUseRmsNormWideTileProjectionFusion(layerWeights, reusesSharedKV) {
  if (hasQ4KMaterialization(layerWeights?.qkvProj)) {
    return true;
  }
  if (!hasQ4KMaterialization(layerWeights?.qProj)) {
    return false;
  }
  if (reusesSharedKV) {
    return true;
  }
  return hasQ4KMaterialization(layerWeights?.kProj)
    && (layerWeights?.vProj == null || hasQ4KMaterialization(layerWeights.vProj));
}
