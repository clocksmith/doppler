/**
 * Reference Gather (Embedding Lookup) Implementation
 */

/**
 * Reference gather/embedding lookup
 * output[i] = embeddings[indices[i]]
 *
 * @param embeddings Embedding table [vocabSize x embedDim]
 * @param indices Token indices [seqLen]
 * @param vocabSize Vocabulary size
 * @param embedDim Embedding dimension
 * @returns Gathered embeddings [seqLen x embedDim]
 */
export function gatherRef(
  embeddings: Float32Array,
  indices: Uint32Array,
  vocabSize: number,
  embedDim: number
): Float32Array {
  const seqLen = indices.length;
  const output = new Float32Array(seqLen * embedDim);

  for (let i = 0; i < seqLen; i++) {
    const idx = indices[i];
    const srcOffset = idx * embedDim;
    const dstOffset = i * embedDim;

    for (let d = 0; d < embedDim; d++) {
      output[dstOffset + d] = embeddings[srcOffset + d];
    }
  }

  return output;
}

/**
 * Batched gather
 * @param embeddings [vocabSize x embedDim]
 * @param indices [batchSize x seqLen]
 * @param batchSize Batch size
 * @param seqLen Sequence length
 * @param embedDim Embedding dimension
 * @returns [batchSize x seqLen x embedDim]
 */
export function batchGatherRef(
  embeddings: Float32Array,
  indices: Uint32Array,
  batchSize: number,
  seqLen: number,
  embedDim: number
): Float32Array {
  const output = new Float32Array(batchSize * seqLen * embedDim);

  for (let b = 0; b < batchSize; b++) {
    for (let s = 0; s < seqLen; s++) {
      const idx = indices[b * seqLen + s];
      const srcOffset = idx * embedDim;
      const dstOffset = (b * seqLen + s) * embedDim;

      for (let d = 0; d < embedDim; d++) {
        output[dstOffset + d] = embeddings[srcOffset + d];
      }
    }
  }

  return output;
}

/**
 * Gather with position embeddings added
 */
export function gatherWithPosRef(
  embeddings: Float32Array,
  posEmbeddings: Float32Array,
  indices: Uint32Array,
  vocabSize: number,
  embedDim: number,
  startPos: number = 0
): Float32Array {
  const seqLen = indices.length;
  const output = new Float32Array(seqLen * embedDim);

  for (let i = 0; i < seqLen; i++) {
    const tokenIdx = indices[i];
    const posIdx = i + startPos;

    const tokenOffset = tokenIdx * embedDim;
    const posOffset = posIdx * embedDim;
    const dstOffset = i * embedDim;

    for (let d = 0; d < embedDim; d++) {
      output[dstOffset + d] = embeddings[tokenOffset + d] + posEmbeddings[posOffset + d];
    }
  }

  return output;
}

export default gatherRef;
