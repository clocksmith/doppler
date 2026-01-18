import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { castF16ToF32 } from '../cast.js';
import { runMatmul } from '../matmul.js';
import { runTranspose } from '../transpose.js';
import { runSoftmaxBackward } from './softmax_backward.js';
import { runBackwardKernel } from './utils.js';

async function ensureF32(tensor) {
  if (tensor.dtype === 'f16') {
    return castF16ToF32(tensor);
  }
  return tensor;
}

async function copySlices(device, slices) {
  const encoder = device.createCommandEncoder();
  const buffers = slices.map((slice) => {
    const buffer = acquireBuffer(slice.size, undefined, slice.label);
    encoder.copyBufferToBuffer(slice.source, slice.offset, buffer, 0, slice.size);
    return buffer;
  });
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return buffers;
}

async function copyInto(device, copies) {
  const encoder = device.createCommandEncoder();
  for (const copy of copies) {
    encoder.copyBufferToBuffer(copy.source, 0, copy.target, copy.offset, copy.size);
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

export async function runAttentionBackward(
  q,
  k,
  v,
  softmax,
  gradOutput,
  options = {}
) {
  const device = getDevice();
  if (!device) {
    throw new Error('runAttentionBackward requires a GPU device');
  }

  const { seqLen, numHeads, headDim, scale = 1.0, causal = false } = options;
  if (!seqLen || !numHeads || !headDim) {
    throw new Error('attention backward requires seqLen, numHeads, and headDim');
  }

  const qTensor = await ensureF32(q);
  const kTensor = await ensureF32(k);
  const vTensor = await ensureF32(v);
  const sTensor = await ensureF32(softmax);
  const dTensor = await ensureF32(gradOutput);

  const headElements = seqLen * headDim;
  const headBytes = headElements * dtypeBytes(qTensor.dtype);
  const softmaxBytes = seqLen * seqLen * dtypeBytes(sTensor.dtype);

  const totalBytes = numHeads * headBytes;
  const gradQBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_q');
  const gradKBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_k');
  const gradVBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_v');

  for (let h = 0; h < numHeads; h += 1) {
    const qOffset = h * headBytes;
    const kOffset = h * headBytes;
    const vOffset = h * headBytes;
    const dOffset = h * headBytes;
    const sOffset = h * softmaxBytes;

    const [qHeadBuf, kHeadBuf, vHeadBuf, sHeadBuf, dHeadBuf] = await copySlices(device, [
      { source: qTensor.buffer, offset: qOffset, size: headBytes, label: 'attn_q_head' },
      { source: kTensor.buffer, offset: kOffset, size: headBytes, label: 'attn_k_head' },
      { source: vTensor.buffer, offset: vOffset, size: headBytes, label: 'attn_v_head' },
      { source: sTensor.buffer, offset: sOffset, size: softmaxBytes, label: 'attn_s_head' },
      { source: dTensor.buffer, offset: dOffset, size: headBytes, label: 'attn_d_head' },
    ]);

    const qHead = createTensor(qHeadBuf, 'f32', [seqLen, headDim], 'attn_q_head');
    const kHead = createTensor(kHeadBuf, 'f32', [seqLen, headDim], 'attn_k_head');
    const vHead = createTensor(vHeadBuf, 'f32', [seqLen, headDim], 'attn_v_head');
    const sHead = createTensor(sHeadBuf, 'f32', [seqLen, seqLen], 'attn_s_head');
    const dHead = createTensor(dHeadBuf, 'f32', [seqLen, headDim], 'attn_d_head');

    const sTransposed = await runTranspose(sHead, seqLen, seqLen);
    const dV = await runMatmul(sTransposed, dHead, seqLen, headDim, seqLen, { transposeB: false });

    const vTransposed = await runTranspose(vHead, seqLen, headDim);
    const dS = await runMatmul(dHead, vTransposed, seqLen, seqLen, headDim, { transposeB: false });
    const dQK = causal
      ? await runBackwardKernel(
        'attention_backward',
        sHead,
        dS,
        16,
        (view) => {
          view.setUint32(0, seqLen, true);
          view.setUint32(4, seqLen, true);
          view.setUint32(8, 1, true);
        }
      )
      : await runSoftmaxBackward(sHead, dS, { rows: seqLen, cols: seqLen });

    const dQ = await runMatmul(dQK, kHead, seqLen, headDim, seqLen, { transposeB: false, alpha: scale });
    const dQKTransposed = await runTranspose(dQK, seqLen, seqLen);
    const dK = await runMatmul(dQKTransposed, qHead, seqLen, headDim, seqLen, { transposeB: false, alpha: scale });

    await copyInto(device, [
      { source: dQ.buffer, target: gradQBuf, offset: qOffset, size: headBytes },
      { source: dK.buffer, target: gradKBuf, offset: kOffset, size: headBytes },
      { source: dV.buffer, target: gradVBuf, offset: vOffset, size: headBytes },
    ]);
  }

  return {
    gradQ: createTensor(gradQBuf, 'f32', [...q.shape], 'attn_grad_q'),
    gradK: createTensor(gradKBuf, 'f32', [...k.shape], 'attn_grad_k'),
    gradV: createTensor(gradVBuf, 'f32', [...v.shape], 'attn_grad_v'),
  };
}

export async function recordAttentionBackward() {
  throw new Error('recordAttentionBackward is not implemented');
}
