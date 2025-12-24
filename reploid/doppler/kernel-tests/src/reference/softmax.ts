/**
 * Reference Softmax Implementation
 */

/**
 * Reference online softmax (numerically stable)
 *
 * @param input Input logits [outerSize x innerSize]
 * @param innerSize Size of softmax dimension
 * @param outerSize Batch/outer dimension
 * @param temperature Temperature scaling
 * @returns Softmax probabilities
 */
export function softmaxRef(
  input: Float32Array,
  innerSize: number,
  outerSize: number,
  temperature: number = 1.0
): Float32Array {
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
 * @param input Input logits
 * @param innerSize Size of softmax dimension
 * @param outerSize Batch/outer dimension
 * @param temperature Temperature scaling
 * @returns Log softmax values
 */
export function logSoftmaxRef(
  input: Float32Array,
  innerSize: number,
  outerSize: number,
  temperature: number = 1.0
): Float32Array {
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
export function softmaxInplaceRef(
  input: Float32Array,
  innerSize: number,
  outerSize: number,
  temperature: number = 1.0
): Float32Array {
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
