export async function ensureExpertLoaded(layerIdx, expertIdx, expertWeights, expertLoader) {
  const key = `layer_${layerIdx}_expert_${expertIdx}`;
  const held = expertWeights.get(key);
  // The loader owns residency: an LRU eviction releases buffers this map still references.
  if (held && expertLoader.isExpertResident(layerIdx, expertIdx, held)) return;
  expertWeights.delete(key);

  const weights = await expertLoader.loadExpert(layerIdx, expertIdx);
  if (weights) {
    expertWeights.set(key, weights);
  }
}
