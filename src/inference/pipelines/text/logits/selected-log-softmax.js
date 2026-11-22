// Exact, deterministic log-softmax of selected tokens over complete logit rows.
// Rows come from Doppler's own LM-head kernels; this is scalar readback math only.

/** @type {typeof import('./selected-log-softmax.js').selectedTokenLogSoftmax} */
export function selectedTokenLogSoftmax(rows, vocabSize, tokenIds) {
  if (!(rows instanceof Float32Array) && !(rows instanceof Float64Array)) {
    throw new Error('[Logits] Log-softmax rows must be a Float32Array or Float64Array.');
  }
  if (!Number.isSafeInteger(vocabSize) || vocabSize < 1) {
    throw new Error('[Logits] Log-softmax requires a positive vocabSize.');
  }
  if (!Array.isArray(tokenIds) || tokenIds.length < 1 || rows.length !== tokenIds.length * vocabSize) {
    throw new Error(`[Logits] Log-softmax expects ${tokenIds?.length ?? 0} complete rows of ${vocabSize} logits.`);
  }
  const logits = [];
  const logSumExp = [];
  const logProbs = [];
  for (let row = 0; row < tokenIds.length; row++) {
    const tokenId = tokenIds[row];
    if (!Number.isSafeInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
      throw new Error(`[Logits] Log-softmax token ${String(tokenId)} is outside [0, ${vocabSize}).`);
    }
    const start = row * vocabSize;
    let max = -Infinity;
    for (let i = start; i < start + vocabSize; i++) {
      const value = rows[i];
      if (Number.isNaN(value) || value === Infinity) {
        throw new Error(`[Logits] Log-softmax row ${row} contains a NaN or +Infinity logit.`);
      }
      if (value > max) max = value;
    }
    if (max === -Infinity) throw new Error(`[Logits] Log-softmax row ${row} has no finite logit.`);
    // Max-subtracted float64 accumulation in index order: stable and reproducible.
    let sum = 0;
    for (let i = start; i < start + vocabSize; i++) sum += Math.exp(rows[i] - max);
    const lse = max + Math.log(sum);
    const logit = rows[start + tokenId];
    logits.push(logit);
    logSumExp.push(lse);
    logProbs.push(logit - lse);
  }
  return { logits, logSumExp, logProbs };
}
