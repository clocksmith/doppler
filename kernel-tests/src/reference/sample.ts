/**
 * Reference Implementation for Sampling Operations
 *
 * These implement CPU-side sampling for validating GPU sample kernels.
 */

/**
 * Argmax - find index of maximum value (greedy decoding)
 */
export function argmaxRef(logits: Float32Array): number {
  let maxIdx = 0;
  let maxVal = logits[0];

  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxVal) {
      maxVal = logits[i];
      maxIdx = i;
    }
  }

  return maxIdx;
}

/**
 * Top-k argmax - find indices of k largest values
 */
export function topkArgmaxRef(logits: Float32Array, k: number): { indices: number[]; values: number[] } {
  // Create index array and sort by value
  const indexed = Array.from(logits).map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => b.val - a.val);

  const topK = indexed.slice(0, k);
  return {
    indices: topK.map(x => x.idx),
    values: topK.map(x => x.val),
  };
}

/**
 * Softmax with temperature
 */
export function softmaxWithTemp(logits: Float32Array, temperature: number): Float32Array {
  const scaled = new Float32Array(logits.length);

  // Apply temperature
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / temperature;
  }

  // Find max for numerical stability
  let max = scaled[0];
  for (let i = 1; i < scaled.length; i++) {
    if (scaled[i] > max) max = scaled[i];
  }

  // Compute exp and sum
  let sum = 0;
  for (let i = 0; i < scaled.length; i++) {
    scaled[i] = Math.exp(scaled[i] - max);
    sum += scaled[i];
  }

  // Normalize
  for (let i = 0; i < scaled.length; i++) {
    scaled[i] /= sum;
  }

  return scaled;
}

/**
 * Top-k sampling with temperature
 * Returns sampled token ID given:
 * - logits: raw logits
 * - temperature: scaling factor
 * - topK: number of candidates
 * - randomValue: pre-generated random [0,1) for reproducibility
 */
export function sampleTopKRef(
  logits: Float32Array,
  temperature: number,
  topK: number,
  randomValue: number
): number {
  // For very low temperature, use greedy
  if (temperature < 0.01) {
    return argmaxRef(logits);
  }

  // Get top-k candidates
  const { indices, values } = topkArgmaxRef(logits, topK);

  // Apply temperature and softmax to top-k values
  const scaledValues = values.map(v => v / temperature);

  // Softmax on scaled values
  const max = Math.max(...scaledValues);
  const expValues = scaledValues.map(v => Math.exp(v - max));
  const sum = expValues.reduce((a, b) => a + b, 0);
  const probs = expValues.map(v => v / sum);

  // Sample from multinomial distribution
  let cumProb = 0;
  for (let i = 0; i < probs.length; i++) {
    cumProb += probs[i];
    if (cumProb >= randomValue) {
      return indices[i];
    }
  }

  // Fallback to last item
  return indices[indices.length - 1];
}

/**
 * Simple seeded random number generator (matches GPU implementation)
 */
export function seededRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}
