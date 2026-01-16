


export function rmsNormRef(input, weight, batchSize, hiddenSize, eps = 1e-6) {
  const output = new Float32Array(input.length);

  for (let b = 0; b < batchSize; b++) {
    const offset = b * hiddenSize;

    // Compute mean of squares
    let sumSq = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const val = input[offset + i];
      sumSq += val * val;
    }
    const meanSq = sumSq / hiddenSize;

    // Compute rsqrt
    const scale = 1.0 / Math.sqrt(meanSq + eps);

    // Apply normalization and weight
    for (let i = 0; i < hiddenSize; i++) {
      output[offset + i] = input[offset + i] * scale * weight[i];
    }
  }

  return output;
}


export function rmsNormNoWeightRef(input, batchSize, hiddenSize, eps = 1e-6) {
  const output = new Float32Array(input.length);

  for (let b = 0; b < batchSize; b++) {
    const offset = b * hiddenSize;

    let sumSq = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const val = input[offset + i];
      sumSq += val * val;
    }
    const scale = 1.0 / Math.sqrt(sumSq / hiddenSize + eps);

    for (let i = 0; i < hiddenSize; i++) {
      output[offset + i] = input[offset + i] * scale;
    }
  }

  return output;
}

export default rmsNormRef;
