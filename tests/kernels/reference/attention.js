


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

export function attentionBackwardRef(Q, K, V, softmax, gradOutput, seqLen, kvLen, numHeads, numKVHeads, headDim, scale) {
  const gradQ = new Float32Array(seqLen * numHeads * headDim);
  const gradK = new Float32Array(kvLen * numKVHeads * headDim);
  const gradV = new Float32Array(kvLen * numKVHeads * headDim);
  
  const headsPerKV = numHeads / numKVHeads;

  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / headsPerKV);

    for (let q = 0; q < seqLen; q++) {
      const qOffset = (q * numHeads + h) * headDim;
      const sOffset = (h * seqLen + q) * kvLen;

      // 1. dV = P^T @ dY
      // softmax P is [seqLen, kvLen] for each head
      // dY is [seqLen, headDim]
      for (let k = 0; k < kvLen; k++) {
        const p = softmax[sOffset + k];
        const vOffset = (k * numKVHeads + kvHead) * headDim;
        for (let d = 0; d < headDim; d++) {
          gradV[vOffset + d] += p * gradOutput[qOffset + d];
        }
      }

      // 2. dP = dY @ V^T
      const dP = new Float32Array(kvLen);
      for (let k = 0; k < kvLen; k++) {
        let sum = 0;
        const vOffset = (k * numKVHeads + kvHead) * headDim;
        for (let d = 0; d < headDim; d++) {
          sum += gradOutput[qOffset + d] * V[vOffset + d];
        }
        dP[k] = sum;
      }

      // 3. dS = SoftmaxBackward(dP)
      // dS = P * (dP - dot(P, dP))
      let dotPP = 0;
      for (let k = 0; k < kvLen; k++) {
        dotPP += softmax[sOffset + k] * dP[k];
      }
      
      const dS = new Float32Array(kvLen);
      for (let k = 0; k < kvLen; k++) {
        dS[k] = softmax[sOffset + k] * (dP[k] - dotPP);
      }

      // 4. dQ = (dS * scale) @ K
      // 5. dK = (dS * scale)^T @ Q
      for (let k = 0; k < kvLen; k++) {
        const dS_scaled = dS[k] * scale;
        const kOffset = (k * numKVHeads + kvHead) * headDim;
        for (let d = 0; d < headDim; d++) {
          gradQ[qOffset + d] += dS_scaled * K[kOffset + d];
          gradK[kOffset + d] += dS_scaled * Q[qOffset + d];
        }
      }
    }
  }

  return { gradQ, gradK, gradV };
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
