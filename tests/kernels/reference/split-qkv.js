


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
