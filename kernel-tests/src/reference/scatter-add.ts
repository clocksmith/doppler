/**
 * Reference Scatter-Add for MoE Output Combination
 */

/**
 * Reference scatter-add for MoE output combination
 * Combines expert outputs weighted by routing probabilities
 *
 * @param expertOutputs Expert outputs [numExperts, numTokens, hiddenSize]
 * @param indices Selected expert indices [numTokens, topK]
 * @param weights Routing weights [numTokens, topK]
 * @param numTokens Number of tokens
 * @param hiddenSize Hidden dimension
 * @param numExperts Number of experts
 * @param topK Top-k experts per token
 * @returns Combined output [numTokens, hiddenSize]
 */
export function scatterAddRef(
  expertOutputs: Float32Array,
  indices: Uint32Array,
  weights: Float32Array,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number
): Float32Array {
  const output = new Float32Array(numTokens * hiddenSize);

  for (let token = 0; token < numTokens; token++) {
    for (let dim = 0; dim < hiddenSize; dim++) {
      let sum = 0;

      for (let k = 0; k < topK; k++) {
        const expertIdx = indices[token * topK + k];
        const weight = weights[token * topK + k];

        // Expert output layout: [numExperts, numTokens, hiddenSize]
        const expertOffset = expertIdx * numTokens * hiddenSize + token * hiddenSize + dim;
        sum += weight * expertOutputs[expertOffset];
      }

      output[token * hiddenSize + dim] = sum;
    }
  }

  return output;
}

/**
 * In-place accumulation variant (adds to existing output)
 */
export function scatterAddAccumulateRef(
  expertOutputs: Float32Array,
  indices: Uint32Array,
  weights: Float32Array,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  existingOutput: Float32Array
): Float32Array {
  const output = new Float32Array(existingOutput);

  for (let token = 0; token < numTokens; token++) {
    for (let dim = 0; dim < hiddenSize; dim++) {
      let sum = 0;

      for (let k = 0; k < topK; k++) {
        const expertIdx = indices[token * topK + k];
        const weight = weights[token * topK + k];
        const expertOffset = expertIdx * numTokens * hiddenSize + token * hiddenSize + dim;
        sum += weight * expertOutputs[expertOffset];
      }

      output[token * hiddenSize + dim] += sum;
    }
  }

  return output;
}

export default scatterAddRef;
