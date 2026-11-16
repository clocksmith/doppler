import { resolveAttentionPlan } from "./attention/plan.js";
import { executeAttention, executeAttentionBDPA } from "./attention/executors.js";
export { runAttentionTiered, recordAttentionTiered } from "./attention/tiered.js";
export { runAttentionTieredQuant, recordAttentionTieredQuant } from "./attention/tiered-quantized.js";
export { runAttentionContiguousQuant, recordAttentionContiguousQuant } from "./attention/contiguous-quantized.js";

export function resolveAttentionPlanForTest(
  seqLen,
  kvLen,
  headDim,
  numHeads,
  kvDtype,
  qDtype,
  sharedLimit,
  caps,
  layerIdx,
  isPaged = false,
  kernelPath = null
) {
  return resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    kvDtype,
    qDtype,
    sharedLimit,
    caps,
    layerIdx,
    isPaged,
    kernelPath
  );
}

export async function runAttentionBDPA(
  Q,
  basisK,
  basisV,
  pagedK,
  pagedV,
  index,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionBDPA(null, Q, basisK, basisV, pagedK, pagedV, index, numHeads, headDim, options);
}

export async function recordAttentionBDPA(
  recorder,
  Q,
  basisK,
  basisV,
  pagedK,
  pagedV,
  index,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionBDPA(recorder, Q, basisK, basisV, pagedK, pagedV, index, numHeads, headDim, options);
}

export async function runAttention(
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttention(null, Q, K, V, mask, numHeads, headDim, options);
}

export async function recordAttention(
  recorder,
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttention(recorder, Q, K, V, mask, numHeads, headDim, options);
}
