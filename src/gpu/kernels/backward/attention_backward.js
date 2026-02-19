import { CommandRecorder } from '../../command-recorder.js';
import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { castF16ToF32, recordCastF16ToF32 } from '../cast.js';
import { runMatmul, recordMatmul } from '../matmul.js';
import { runTranspose, recordTranspose } from '../transpose.js';
import { runSoftmaxBackward, recordSoftmaxBackward } from './softmax_backward.js';
import { runBackwardKernel, recordBackwardKernel } from './utils.js';

async function ensureF32(tensor, recorder = null) {
  if (tensor.dtype !== 'f16') {
    return tensor;
  }
  if (!recorder) {
    return castF16ToF32(tensor);
  }
  const casted = await recordCastF16ToF32(recorder, tensor);
  recorder.trackTemporaryBuffer(casted.buffer);
  return casted;
}

function createHeadSliceBuffers(recorder, headBytes, softmaxBytes) {
  const qHeadBuf = acquireBuffer(headBytes, undefined, 'attn_q_head');
  const kHeadBuf = acquireBuffer(headBytes, undefined, 'attn_k_head');
  const vHeadBuf = acquireBuffer(headBytes, undefined, 'attn_v_head');
  const sHeadBuf = acquireBuffer(softmaxBytes, undefined, 'attn_s_head');
  const dHeadBuf = acquireBuffer(headBytes, undefined, 'attn_d_head');

  recorder.trackTemporaryBuffer(qHeadBuf);
  recorder.trackTemporaryBuffer(kHeadBuf);
  recorder.trackTemporaryBuffer(vHeadBuf);
  recorder.trackTemporaryBuffer(sHeadBuf);
  recorder.trackTemporaryBuffer(dHeadBuf);

  return { qHeadBuf, kHeadBuf, vHeadBuf, sHeadBuf, dHeadBuf };
}

function createHeadTensors(qHeadBuf, kHeadBuf, vHeadBuf, sHeadBuf, dHeadBuf, seqLen, headDim) {
  const qHead = createTensor(qHeadBuf, 'f32', [seqLen, headDim], 'attn_q_head');
  const kHead = createTensor(kHeadBuf, 'f32', [seqLen, headDim], 'attn_k_head');
  const vHead = createTensor(vHeadBuf, 'f32', [seqLen, headDim], 'attn_v_head');
  const sHead = createTensor(sHeadBuf, 'f32', [seqLen, seqLen], 'attn_s_head');
  const dHead = createTensor(dHeadBuf, 'f32', [seqLen, headDim], 'attn_d_head');
  return { qHead, kHead, vHead, sHead, dHead };
}

function trackTensorBuffer(recorder, tensor) {
  recorder.trackTemporaryBuffer(tensor.buffer);
}

async function runAttentionBackwardCore(
  q,
  k,
  v,
  softmax,
  gradOutput,
  options = {},
  recorder = null
) {
  const { seqLen, numHeads, headDim, scale = 1.0, causal = false } = options;
  if (!seqLen || !numHeads || !headDim) {
    throw new Error('attention backward requires seqLen, numHeads, and headDim');
  }

  const qTensor = await ensureF32(q, recorder);
  const kTensor = await ensureF32(k, recorder);
  const vTensor = await ensureF32(v, recorder);
  const sTensor = await ensureF32(softmax, recorder);
  const dTensor = await ensureF32(gradOutput, recorder);

  const headElements = seqLen * headDim;
  const headBytes = headElements * dtypeBytes(qTensor.dtype);
  const softmaxBytes = seqLen * seqLen * dtypeBytes(sTensor.dtype);

  const totalBytes = numHeads * headBytes;
  const gradQBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_q');
  const gradKBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_k');
  const gradVBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_v');

  if (!recorder) {
    for (let h = 0; h < numHeads; h += 1) {
      const qOffset = h * headBytes;
      const kOffset = h * headBytes;
      const vOffset = h * headBytes;
      const dOffset = h * headBytes;
      const sOffset = h * softmaxBytes;

      const qHeadBuf = acquireBuffer(headBytes, undefined, 'attn_q_head');
      const kHeadBuf = acquireBuffer(headBytes, undefined, 'attn_k_head');
      const vHeadBuf = acquireBuffer(headBytes, undefined, 'attn_v_head');
      const sHeadBuf = acquireBuffer(softmaxBytes, undefined, 'attn_s_head');
      const dHeadBuf = acquireBuffer(headBytes, undefined, 'attn_d_head');

      const sliceEncoder = getDevice().createCommandEncoder();
      sliceEncoder.copyBufferToBuffer(qTensor.buffer, qOffset, qHeadBuf, 0, headBytes);
      sliceEncoder.copyBufferToBuffer(kTensor.buffer, kOffset, kHeadBuf, 0, headBytes);
      sliceEncoder.copyBufferToBuffer(vTensor.buffer, vOffset, vHeadBuf, 0, headBytes);
      sliceEncoder.copyBufferToBuffer(sTensor.buffer, sOffset, sHeadBuf, 0, softmaxBytes);
      sliceEncoder.copyBufferToBuffer(dTensor.buffer, dOffset, dHeadBuf, 0, headBytes);
      getDevice().queue.submit([sliceEncoder.finish()]);

      const { qHead, kHead, vHead, sHead, dHead } = createHeadTensors(
        qHeadBuf,
        kHeadBuf,
        vHeadBuf,
        sHeadBuf,
        dHeadBuf,
        seqLen,
        headDim
      );

      const sTransposed = await runTranspose(sHead, seqLen, seqLen);
      const dV = await runMatmul(sTransposed, dHead.buffer, seqLen, headDim, seqLen, {
        transposeB: false,
        bDtype: 'f32',
      });

      const vTransposed = await runTranspose(vHead, seqLen, headDim);
      const dS = await runMatmul(dHead, vTransposed.buffer, seqLen, seqLen, headDim, {
        transposeB: false,
        bDtype: 'f32',
      });
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

      const dQ = await runMatmul(dQK, kHead.buffer, seqLen, headDim, seqLen, {
        transposeB: false,
        alpha: scale,
        bDtype: 'f32',
      });
      const dQKTransposed = await runTranspose(dQK, seqLen, seqLen);
      const dK = await runMatmul(dQKTransposed, qHead.buffer, seqLen, headDim, seqLen, {
        transposeB: false,
        alpha: scale,
        bDtype: 'f32',
      });

      const copyEncoder = getDevice().createCommandEncoder();
      copyEncoder.copyBufferToBuffer(dQ.buffer, 0, gradQBuf, qOffset, headBytes);
      copyEncoder.copyBufferToBuffer(dK.buffer, 0, gradKBuf, kOffset, headBytes);
      copyEncoder.copyBufferToBuffer(dV.buffer, 0, gradVBuf, vOffset, headBytes);
      getDevice().queue.submit([copyEncoder.finish()]);
    }
  } else {
    const encoder = recorder.getEncoder();
    for (let h = 0; h < numHeads; h += 1) {
      const qOffset = h * headBytes;
      const kOffset = h * headBytes;
      const vOffset = h * headBytes;
      const dOffset = h * headBytes;
      const sOffset = h * softmaxBytes;

      const { qHeadBuf, kHeadBuf, vHeadBuf, sHeadBuf, dHeadBuf } = createHeadSliceBuffers(
        recorder,
        headBytes,
        softmaxBytes
      );

      encoder.copyBufferToBuffer(qTensor.buffer, qOffset, qHeadBuf, 0, headBytes);
      encoder.copyBufferToBuffer(kTensor.buffer, kOffset, kHeadBuf, 0, headBytes);
      encoder.copyBufferToBuffer(vTensor.buffer, vOffset, vHeadBuf, 0, headBytes);
      encoder.copyBufferToBuffer(sTensor.buffer, sOffset, sHeadBuf, 0, softmaxBytes);
      encoder.copyBufferToBuffer(dTensor.buffer, dOffset, dHeadBuf, 0, headBytes);

      const { qHead, kHead, vHead, sHead, dHead } = createHeadTensors(
        qHeadBuf,
        kHeadBuf,
        vHeadBuf,
        sHeadBuf,
        dHeadBuf,
        seqLen,
        headDim
      );

      const sTransposed = await recordTranspose(recorder, sHead, seqLen, seqLen);
      const dV = await recordMatmul(recorder, sTransposed, dHead.buffer, seqLen, headDim, seqLen, {
        transposeB: false,
        bDtype: 'f32',
      });

      const vTransposed = await recordTranspose(recorder, vHead, seqLen, headDim);
      const dS = await recordMatmul(recorder, dHead, vTransposed.buffer, seqLen, seqLen, headDim, {
        transposeB: false,
        bDtype: 'f32',
      });
      const dQK = causal
        ? await recordBackwardKernel(
          recorder,
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
        : await recordSoftmaxBackward(recorder, sHead, dS, { rows: seqLen, cols: seqLen });

      const dQ = await recordMatmul(recorder, dQK, kHead.buffer, seqLen, headDim, seqLen, {
        transposeB: false,
        alpha: scale,
        bDtype: 'f32',
      });
      const dQKTransposed = await recordTranspose(recorder, dQK, seqLen, seqLen);
      const dK = await recordMatmul(recorder, dQKTransposed, qHead.buffer, seqLen, headDim, seqLen, {
        transposeB: false,
        alpha: scale,
        bDtype: 'f32',
      });

      encoder.copyBufferToBuffer(dQ.buffer, 0, gradQBuf, qOffset, headBytes);
      encoder.copyBufferToBuffer(dK.buffer, 0, gradKBuf, kOffset, headBytes);
      encoder.copyBufferToBuffer(dV.buffer, 0, gradVBuf, vOffset, headBytes);

      trackTensorBuffer(recorder, sTransposed);
      trackTensorBuffer(recorder, dV);
      trackTensorBuffer(recorder, vTransposed);
      trackTensorBuffer(recorder, dS);
      trackTensorBuffer(recorder, dQK);
      trackTensorBuffer(recorder, dQ);
      trackTensorBuffer(recorder, dQKTransposed);
      trackTensorBuffer(recorder, dK);
    }
  }

  return {
    gradQ: createTensor(gradQBuf, 'f32', [...q.shape], 'attn_grad_q'),
    gradK: createTensor(gradKBuf, 'f32', [...k.shape], 'attn_grad_k'),
    gradV: createTensor(gradVBuf, 'f32', [...v.shape], 'attn_grad_v'),
  };
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

  const recorder = new CommandRecorder(device, 'attention_backward');
  const result = await runAttentionBackwardCore(q, k, v, softmax, gradOutput, options, recorder);
  recorder.submit();
  return result;
}

export async function recordAttentionBackward(
  recorder,
  q,
  k,
  v,
  softmax,
  gradOutput,
  options = {}
) {
  return runAttentionBackwardCore(q, k, v, softmax, gradOutput, options, recorder);
}
