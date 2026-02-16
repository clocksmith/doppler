export async function ensureExpertLoaded(layerIdx, expertIdx, expertWeights, expertLoader) {
  const key = `layer_${layerIdx}_expert_${expertIdx}`;
  if (expertWeights.has(key)) return;

  const weights = await expertLoader.loadExpert(layerIdx, expertIdx);
  if (weights) {
    expertWeights.set(key, weights);
  }
}

export function gatherTokens(hiddenStates, indices, hiddenSize) {
  const gathered = new Float32Array(indices.length * hiddenSize);
  for (let i = 0; i < indices.length; i++) {
    const srcOffset = indices[i] * hiddenSize;
    gathered.set(
      hiddenStates.subarray(srcOffset, srcOffset + hiddenSize),
      i * hiddenSize
    );
  }
  return gathered;
}
