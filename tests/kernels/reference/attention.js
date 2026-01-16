


export function attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, mask = null) {
  const output = new Float32Array(seqLen * numHeads * headDim);
  const scale = 1.0 / Math.sqrt(headDim);

  // Number of query heads per KV head (for GQA)
  const headsPerKV = numHeads / numKVHeads;

  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / headsPerKV);

    for (let q = 0; q < seqLen; q++) {
      // Compute attention scores for this query position
      const scores = new Float32Array(kvLen);

      // Q @ K^T
      for (let k = 0; k < kvLen; k++) {
        let score = 0;
        for (let d = 0; d < headDim; d++) {
          const qIdx = q * numHeads * headDim + h * headDim + d;
          const kIdx = k * numKVHeads * headDim + kvHead * headDim + d;
          score += Q[qIdx] * K[kIdx];
        }
        scores[k] = score * scale;

        // Apply mask if provided
        if (mask) {
          scores[k] += mask[q * kvLen + k];
        }
      }

      // Softmax
      let maxScore = -Infinity;
      for (let k = 0; k < kvLen; k++) {
        maxScore = Math.max(maxScore, scores[k]);
      }

      let sumExp = 0;
      for (let k = 0; k < kvLen; k++) {
        scores[k] = Math.exp(scores[k] - maxScore);
        sumExp += scores[k];
      }

      for (let k = 0; k < kvLen; k++) {
        scores[k] /= sumExp;
      }

      // Attention @ V
      for (let d = 0; d < headDim; d++) {
        let val = 0;
        for (let k = 0; k < kvLen; k++) {
          const vIdx = k * numKVHeads * headDim + kvHead * headDim + d;
          val += scores[k] * V[vIdx];
        }
        output[q * numHeads * headDim + h * headDim + d] = val;
      }
    }
  }

  return output;
}


export function createCausalMask(seqLen, kvLen = null) {
  if (kvLen === null) kvLen = seqLen;

  const mask = new Float32Array(seqLen * kvLen);

  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < kvLen; j++) {
      // For causal: can attend to positions <= current
      // Offset by (kvLen - seqLen) for KV cache scenarios
      const offset = kvLen - seqLen;
      mask[i * kvLen + j] = j <= i + offset ? 0 : -Infinity;
    }
  }

  return mask;
}


export function flashAttentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, blockSize = 64) {
  // This is just a reference that produces the same result
  // Real flash attention saves memory by not materializing full attention matrix
  return attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, createCausalMask(seqLen, kvLen));
}


export function mqaRef(Q, K, V, seqLen, kvLen, numHeads, headDim, mask = null) {
  return attentionRef(Q, K, V, seqLen, kvLen, numHeads, 1, headDim, mask);
}

export default attentionRef;
