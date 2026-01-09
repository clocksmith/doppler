/**
 * Split QKV Reference Implementation
 *
 * Splits a fused QKV tensor into separate Q, K, V tensors.
 *
 * Input layout (row-major):
 *   [numTokens, qSize + kSize + vSize]
 *   Each row: [q_values, k_values, v_values]
 *
 * Output layout:
 *   Q: [numTokens, qSize]
 *   K: [numTokens, kSize]
 *   V: [numTokens, vSize]
 */

/**
 * Split fused QKV tensor into separate Q, K, V
 * @param {Float32Array} qkv - Fused QKV tensor [numTokens, qSize + kSize + vSize]
 * @param {number} numTokens - Number of tokens
 * @param {number} qSize - Size of Q per token
 * @param {number} kSize - Size of K per token
 * @param {number} vSize - Size of V per token
 * @returns {{Q: Float32Array, K: Float32Array, V: Float32Array}}
 */
export function splitQkvRef(qkv, numTokens, qSize, kSize, vSize) {
  const qkvSize = qSize + kSize + vSize;

  const Q = new Float32Array(numTokens * qSize);
  const K = new Float32Array(numTokens * kSize);
  const V = new Float32Array(numTokens * vSize);

  for (let t = 0; t < numTokens; t++) {
    const srcOffset = t * qkvSize;

    // Copy Q values
    for (let i = 0; i < qSize; i++) {
      Q[t * qSize + i] = qkv[srcOffset + i];
    }

    // Copy K values
    for (let i = 0; i < kSize; i++) {
      K[t * kSize + i] = qkv[srcOffset + qSize + i];
    }

    // Copy V values
    for (let i = 0; i < vSize; i++) {
      V[t * vSize + i] = qkv[srcOffset + qSize + kSize + i];
    }
  }

  return { Q, K, V };
}

/**
 * Create a fused QKV tensor from separate Q, K, V tensors
 * Useful for creating test data
 * @param {Float32Array} Q - Q tensor [numTokens, qSize]
 * @param {Float32Array} K - K tensor [numTokens, kSize]
 * @param {Float32Array} V - V tensor [numTokens, vSize]
 * @param {number} numTokens - Number of tokens
 * @param {number} qSize - Size of Q per token
 * @param {number} kSize - Size of K per token
 * @param {number} vSize - Size of V per token
 * @returns {Float32Array}
 */
export function fuseQkvRef(Q, K, V, numTokens, qSize, kSize, vSize) {
  const qkvSize = qSize + kSize + vSize;
  const qkv = new Float32Array(numTokens * qkvSize);

  for (let t = 0; t < numTokens; t++) {
    const dstOffset = t * qkvSize;

    // Copy Q values
    for (let i = 0; i < qSize; i++) {
      qkv[dstOffset + i] = Q[t * qSize + i];
    }

    // Copy K values
    for (let i = 0; i < kSize; i++) {
      qkv[dstOffset + qSize + i] = K[t * kSize + i];
    }

    // Copy V values
    for (let i = 0; i < vSize; i++) {
      qkv[dstOffset + qSize + kSize + i] = V[t * vSize + i];
    }
  }

  return qkv;
}
