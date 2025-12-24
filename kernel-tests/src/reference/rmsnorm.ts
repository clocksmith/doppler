/**
 * Reference RMSNorm Implementation
 */

/**
 * Reference RMSNorm
 * y = x * rsqrt(mean(x^2) + eps) * weight
 *
 * @param input Input [batchSize x hiddenSize]
 * @param weight Scale weights [hiddenSize]
 * @param batchSize Number of sequences
 * @param hiddenSize Hidden dimension
 * @param eps Epsilon for numerical stability
 * @returns Normalized output
 */
export function rmsNormRef(
  input: Float32Array,
  weight: Float32Array,
  batchSize: number,
  hiddenSize: number,
  eps: number = 1e-6
): Float32Array {
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

/**
 * RMSNorm without learned weights (just normalization)
 */
export function rmsNormNoWeightRef(
  input: Float32Array,
  batchSize: number,
  hiddenSize: number,
  eps: number = 1e-6
): Float32Array {
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
