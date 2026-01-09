/**
 * Reference Softmax Implementation
 */

/**
 * Reference online softmax (numerically stable)
 */
export function softmaxRef(input, innerSize, outerSize, temperature = 1.0) {
  const output = new Float32Array(input.length);

  for (let row = 0; row < outerSize; row++) {
    const offset = row * innerSize;

    // Find max for numerical stability
    let maxVal = -Infinity;
    for (let i = 0; i < innerSize; i++) {
      maxVal = Math.max(maxVal, input[offset + i] / temperature);
    }

    // Compute exp and sum
    let sum = 0;
    for (let i = 0; i < innerSize; i++) {
      const expVal = Math.exp(input[offset + i] / temperature - maxVal);
      output[offset + i] = expVal;
      sum += expVal;
    }

    // Normalize
    for (let i = 0; i < innerSize; i++) {
      output[offset + i] /= sum;
    }
  }

  return output;
}

/**
 * Log softmax (useful for cross-entropy loss)
 */
export function logSoftmaxRef(input, innerSize, outerSize, temperature = 1.0) {
  const output = new Float32Array(input.length);

  for (let row = 0; row < outerSize; row++) {
    const offset = row * innerSize;

    // Find max
    let maxVal = -Infinity;
    for (let i = 0; i < innerSize; i++) {
      maxVal = Math.max(maxVal, input[offset + i] / temperature);
    }

    // Compute log(sum(exp))
    let logSum = 0;
    for (let i = 0; i < innerSize; i++) {
      logSum += Math.exp(input[offset + i] / temperature - maxVal);
    }
    logSum = Math.log(logSum);

    // log_softmax = x - max - log(sum(exp(x - max)))
    for (let i = 0; i < innerSize; i++) {
      output[offset + i] = input[offset + i] / temperature - maxVal - logSum;
    }
  }

  return output;
}

/**
 * Softmax in-place (modifies input array)
 */
export function softmaxInplaceRef(input, innerSize, outerSize, temperature = 1.0) {
  for (let row = 0; row < outerSize; row++) {
    const offset = row * innerSize;

    let maxVal = -Infinity;
    for (let i = 0; i < innerSize; i++) {
      maxVal = Math.max(maxVal, input[offset + i] / temperature);
    }

    let sum = 0;
    for (let i = 0; i < innerSize; i++) {
      input[offset + i] = Math.exp(input[offset + i] / temperature - maxVal);
      sum += input[offset + i];
    }

    for (let i = 0; i < innerSize; i++) {
      input[offset + i] /= sum;
    }
  }

  return input;
}

export default softmaxRef;
