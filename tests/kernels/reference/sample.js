


export function argmaxRef(logits) {
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


export function topkArgmaxRef(logits, k) {
  // Create index array and sort by value
  const indexed = Array.from(logits).map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => b.val - a.val);

  const topK = indexed.slice(0, k);
  return {
    indices: topK.map(x => x.idx),
    values: topK.map(x => x.val),
  };
}


export function softmaxWithTemp(logits, temperature) {
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


export function sampleTopKRef(logits, temperature, topK, randomValue) {
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


export function seededRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}
