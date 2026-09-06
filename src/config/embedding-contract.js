// Shared signed-manifest admission for Forge qualification and Capsule invocation.
export function resolveCapsuleEmbeddingContract(manifest) {
  if (manifest.modelType !== 'embedding' && manifest.inference?.supportsEmbedding !== true) {
    throw new Error('Capsule manifest does not declare text embedding support.');
  }
  const postprocessor = manifest.inference?.output?.embeddingPostprocessor;
  if (!postprocessor || !['mean', 'last'].includes(postprocessor.poolingMode)
    || postprocessor.includePrompt !== true || !Array.isArray(postprocessor.projections)
    || (postprocessor.normalize !== null && postprocessor.normalize !== 'l2')) {
    throw new Error('Capsule embed requires an explicit supported embeddingPostprocessor; runtime pooling defaults are not permitted.');
  }
  let dimension = manifest.architecture?.hiddenSize;
  if (!Number.isSafeInteger(dimension) || dimension < 1) {
    throw new Error('Capsule embed requires manifest.architecture.hiddenSize.');
  }
  for (const projection of postprocessor.projections) {
    if (projection?.inputSize !== dimension || !Number.isSafeInteger(projection.outputSize)
      || projection.outputSize < 1 || projection.activation !== 'identity'
      || typeof projection.weightTensor !== 'string' || !projection.weightTensor.trim()
      || (projection.biasTensor !== null && (typeof projection.biasTensor !== 'string' || !projection.biasTensor.trim()))) {
      throw new Error('Capsule embed projection geometry or semantics are missing or inconsistent.');
    }
    dimension = projection.outputSize;
  }
  return { dimension, postprocessor: structuredClone(postprocessor) };
}
