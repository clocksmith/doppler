import { runAttention } from '../../../gpu/kernels/attention.js';
import { runSplitQKV } from '../../../gpu/kernels/split_qkv.js';
import { createTensor } from '../../../gpu/tensor.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';

export async function computeVisionAttention(params) {
  const { qkv, seqLen, numHeads, headDim, hiddenSize } = params;
  if (hiddenSize !== numHeads * headDim) {
    throw new Error(
      `Vision attention geometry mismatch: hiddenSize=${hiddenSize}, ` +
      `numHeads=${numHeads}, headDim=${headDim}.`
    );
  }

  const qkvTensor = createTensor(qkv, 'f32', [seqLen, 3 * hiddenSize], 'vision_qkv');
  const split = await runSplitQKV(qkvTensor, {
    numTokens: seqLen,
    qSize: hiddenSize,
    kSize: hiddenSize,
    vSize: hiddenSize,
  });
  let output = null;
  try {
    output = await runAttention(
      createTensor(split.Q.buffer, 'f32', [seqLen, numHeads, headDim], 'vision_q'),
      createTensor(split.K.buffer, 'f32', [seqLen, numHeads, headDim], 'vision_k'),
      createTensor(split.V.buffer, 'f32', [seqLen, numHeads, headDim], 'vision_v'),
      null,
      numHeads,
      headDim,
      {
        seqLen,
        kvLen: seqLen,
        numKVHeads: numHeads,
        scale: 1 / Math.sqrt(headDim),
        causal: false,
        bidirectionalSpanStart: 0,
        bidirectionalSpanLength: 0,
        startPos: 0,
        outputBuffer: null,
        attnSoftcap: 0,
        slidingWindow: 0,
        kvLenBuffer: null,
        kvStart: 0,
        kvLayout: 'contiguous',
        kvPageTable: null,
        kvPageSize: 0,
        indirectBuffer: null,
        kernelPath: null,
        outputGate: null,
      }
    );
    return output.buffer;
  } finally {
    releaseBuffer(split.Q.buffer);
    releaseBuffer(split.K.buffer);
    releaseBuffer(split.V.buffer);
  }
}
