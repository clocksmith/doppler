export function poolSequenceTokenEmbeddings(
  tokenEmbeddings,
  tokenIds,
  hiddenSize,
  pooling
) {
  if (!(tokenEmbeddings instanceof Float32Array)) {
    throw new Error('Sequence token embeddings are unavailable.');
  }
  if (tokenEmbeddings.length !== tokenIds.length * hiddenSize) {
    throw new Error(
      `Sequence embedding shape mismatch: expected ${tokenIds.length * hiddenSize}, got ${tokenEmbeddings.length}.`
    );
  }
  const excluded = new Set(pooling.excludeTokenIds);
  const includedIndices = [];
  for (let index = 0; index < tokenIds.length; index += 1) {
    if (!excluded.has(tokenIds[index])) includedIndices.push(index);
  }
  if (includedIndices.length === 0) {
    throw new Error('Sequence pooling excluded every input token.');
  }
  const pooled = new Float32Array(hiddenSize);
  if (pooling.mode === 'last') {
    const tokenIndex = includedIndices[includedIndices.length - 1];
    const offset = tokenIndex * hiddenSize;
    pooled.set(tokenEmbeddings.subarray(offset, offset + hiddenSize));
  } else {
    for (const tokenIndex of includedIndices) {
      const offset = tokenIndex * hiddenSize;
      for (let column = 0; column < hiddenSize; column += 1) {
        pooled[column] += tokenEmbeddings[offset + column];
      }
    }
    const scale = 1 / includedIndices.length;
    for (let column = 0; column < hiddenSize; column += 1) {
      pooled[column] *= scale;
    }
  }
  return {
    pooled,
    tokenMask: Uint8Array.from(tokenIds, (tokenId) => excluded.has(tokenId) ? 0 : 1),
    includedTokenCount: includedIndices.length,
  };
}
