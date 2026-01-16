


export function gatherRef(embeddings, indices, vocabSize, embedDim) {
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


export function batchGatherRef(embeddings, indices, batchSize, seqLen, embedDim) {
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


export function gatherWithPosRef(embeddings, posEmbeddings, indices, vocabSize, embedDim, startPos = 0) {
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
