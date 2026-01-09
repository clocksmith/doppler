/**
 * Reference Top-K Selection for MoE Routing
 */

/**
 * Reference top-k selection
 */
export function topkRef(probs, numTokens, numExperts, topK, normalize = true) {
  const indices = new Uint32Array(numTokens * topK);
  const weights = new Float32Array(numTokens * topK);

  for (let token = 0; token < numTokens; token++) {
    const offset = token * numExperts;

    // Extract probabilities with indices
    const pairs = [];
    for (let i = 0; i < numExperts; i++) {
      pairs.push({ prob: probs[offset + i], idx: i });
    }

    // Sort descending by probability
    pairs.sort((a, b) => b.prob - a.prob);

    // Take top-k
    let weightSum = 0;
    for (let k = 0; k < topK; k++) {
      indices[token * topK + k] = pairs[k].idx;
      weights[token * topK + k] = pairs[k].prob;
      weightSum += pairs[k].prob;
    }

    // Renormalize if requested
    if (normalize && weightSum > 0) {
      for (let k = 0; k < topK; k++) {
        weights[token * topK + k] /= weightSum;
      }
    }
  }

  return { indices, weights };
}

/**
 * Combined softmax + top-k (reference for fused kernel)
 */
export function softmaxTopkRef(logits, numTokens, numExperts, topK, normalize = true) {
  const indices = new Uint32Array(numTokens * topK);
  const weights = new Float32Array(numTokens * topK);

  for (let token = 0; token < numTokens; token++) {
    const offset = token * numExperts;

    // Find max for numerical stability
    let maxVal = -Infinity;
    for (let i = 0; i < numExperts; i++) {
      maxVal = Math.max(maxVal, logits[offset + i]);
    }

    // Compute exp and sum
    const expVals = new Float32Array(numExperts);
    let expSum = 0;
    for (let i = 0; i < numExperts; i++) {
      expVals[i] = Math.exp(logits[offset + i] - maxVal);
      expSum += expVals[i];
    }

    // Normalize to get probabilities
    const pairs = [];
    for (let i = 0; i < numExperts; i++) {
      pairs.push({ prob: expVals[i] / expSum, idx: i });
    }

    // Sort descending
    pairs.sort((a, b) => b.prob - a.prob);

    // Take top-k and optionally renormalize
    let weightSum = 0;
    for (let k = 0; k < topK; k++) {
      indices[token * topK + k] = pairs[k].idx;
      weights[token * topK + k] = pairs[k].prob;
      weightSum += pairs[k].prob;
    }

    if (normalize && weightSum > 0) {
      for (let k = 0; k < topK; k++) {
        weights[token * topK + k] /= weightSum;
      }
    }
  }

  return { indices, weights };
}

export default topkRef;
