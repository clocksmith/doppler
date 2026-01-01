/**
 * Reference Attention Implementation
 */

/**
 * Reference scaled dot-product attention
 * Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * @param Q Queries [seqLen, numHeads, headDim]
 * @param K Keys [kvLen, numKVHeads, headDim]
 * @param V Values [kvLen, numKVHeads, headDim]
 * @param seqLen Query sequence length
 * @param kvLen Key/Value sequence length
 * @param numHeads Number of query heads
 * @param numKVHeads Number of KV heads (for GQA)
 * @param headDim Dimension per head
 * @param mask Optional attention mask [seqLen, kvLen]
 * @returns Output [seqLen, numHeads, headDim]
 */
export function attentionRef(
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  seqLen: number,
  kvLen: number,
  numHeads: number,
  numKVHeads: number,
  headDim: number,
  mask: Float32Array | null = null
): Float32Array {
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

/**
 * Create causal attention mask
 * Returns mask where mask[i,j] = 0 if j <= i, else -inf
 */
export function createCausalMask(seqLen: number, kvLen: number | null = null): Float32Array {
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

/**
 * Flash attention style - fused attention with chunked computation
 * (Reference only - actual flash attention is GPU-specific)
 */
export function flashAttentionRef(
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  seqLen: number,
  kvLen: number,
  numHeads: number,
  numKVHeads: number,
  headDim: number,
  blockSize: number = 64
): Float32Array {
  // This is just a reference that produces the same result
  // Real flash attention saves memory by not materializing full attention matrix
  return attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, createCausalMask(seqLen, kvLen));
}

/**
 * Multi-query attention (all heads share same K,V)
 */
export function mqaRef(
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  seqLen: number,
  kvLen: number,
  numHeads: number,
  headDim: number,
  mask: Float32Array | null = null
): Float32Array {
  return attentionRef(Q, K, V, seqLen, kvLen, numHeads, 1, headDim, mask);
}

export default attentionRef;
