import { runAttention } from '../gpu/kernels/index.js';
import { buildAttentionSoftmaxCache } from './attention-backward.js';
import { OpType } from './autograd.js';

export async function recordAttentionForward(
  q,
  k,
  v,
  config,
  tape,
  options = {}
) {
  if (!tape || typeof tape.record !== 'function') {
    throw new Error('recordAttentionForward requires an autograd tape');
  }

  const {
    seqLen,
    numHeads,
    numKVHeads = numHeads,
    headDim,
    scale,
    causal = true,
    startPos = 0,
    attnSoftcap = 0,
    slidingWindow = 0,
    kvLen = seqLen,
  } = options;

  if (!seqLen || !numHeads || !headDim) {
    throw new Error('recordAttentionForward requires seqLen, numHeads, and headDim');
  }

  const resolvedScale = scale ?? 1.0 / Math.sqrt(headDim);
  const recomputeForward = Boolean(config?.training?.attention?.recomputeForward);
  const softmax = recomputeForward
    ? null
    : await buildAttentionSoftmaxCache(q, k, {
      seqLen,
      numHeads,
      headDim,
      scale: resolvedScale,
      causal,
    });

  const attentionOptions = {
    seqLen,
    kvLen,
    numHeads,
    numKVHeads,
    headDim,
    scale: resolvedScale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
  };

  const output = await tape.record(
    OpType.ATTENTION,
    (inputQ, inputK, inputV) => runAttention(
      inputQ,
      inputK,
      inputV,
      null,
      numHeads,
      headDim,
      attentionOptions
    ),
    [q, k, v, softmax],
    {
      seqLen,
      numHeads,
      headDim,
      scale: resolvedScale,
      causal,
      recomputeForward,
    }
  );

  return { output, softmax };
}
