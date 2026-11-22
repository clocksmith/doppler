/**
 * Natural-log softmax probability of `tokenIds[r]` in row `r` of `rows` ([tokenIds.length, vocabSize]).
 * Computed over the complete row with max subtraction and float64 accumulation in index order.
 * A selected token whose logit is -Infinity yields -Infinity; NaN, +Infinity or all-masked rows throw.
 */
export function selectedTokenLogSoftmax(
  rows: Float32Array | Float64Array,
  vocabSize: number,
  tokenIds: number[]
): { logits: number[]; logSumExp: number[]; logProbs: number[] };
