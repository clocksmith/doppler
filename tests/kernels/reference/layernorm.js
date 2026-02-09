export function layerNormRef(data, weight, bias, batchSize, hiddenSize, eps) {
  const output = new Float32Array(batchSize * hiddenSize);
  for (let b = 0; b < batchSize; b++) {
    const base = b * hiddenSize;
    let mean = 0;
    for (let i = 0; i < hiddenSize; i++) {
      mean += data[base + i];
    }
    mean /= hiddenSize;
    let varSum = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const diff = data[base + i] - mean;
      varSum += diff * diff;
    }
    const invStd = 1 / Math.sqrt(varSum / hiddenSize + eps);
    for (let i = 0; i < hiddenSize; i++) {
      const norm = (data[base + i] - mean) * invStd;
      output[base + i] = norm * weight[i] + bias[i];
    }
  }
  return output;
}
