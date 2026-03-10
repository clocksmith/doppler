import { getDevice } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
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
  return recordCastF16ToF32(recorder, tensor);
}

function createHeadSliceBuffers(headBytes, softmaxBytes) {
  const qHeadBuf = acquireBuffer(headBytes, undefined, 'attn_q_head');
  const kHeadBuf = acquireBuffer(headBytes, undefined, 'attn_k_head');
  const vHeadBuf = acquireBuffer(headBytes, undefined, 'attn_v_head');
  const sHeadBuf = acquireBuffer(softmaxBytes, undefined, 'attn_s_head');
  const dHeadBuf = acquireBuffer(headBytes, undefined, 'attn_d_head');

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

function releaseTensorBuffer(tensor) {
  if (tensor?.buffer) {
    releaseBuffer(tensor.buffer);
  }
}

function maybeTrackOwnedTensor(ownedTensors, originalTensor, resolvedTensor) {
  if (resolvedTensor !== originalTensor) {
    ownedTensors.push(resolvedTensor);
  }
  return resolvedTensor;
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

  const ownedInputTensors = [];
  const ownedRecorderInputTensors = [];
  const qTensor = !recorder
    ? maybeTrackOwnedTensor(ownedInputTensors, q, await ensureF32(q))
    : maybeTrackOwnedTensor(ownedRecorderInputTensors, q, await ensureF32(q, recorder));
  const kTensor = !recorder
    ? maybeTrackOwnedTensor(ownedInputTensors, k, await ensureF32(k))
    : maybeTrackOwnedTensor(ownedRecorderInputTensors, k, await ensureF32(k, recorder));
  const vTensor = !recorder
    ? maybeTrackOwnedTensor(ownedInputTensors, v, await ensureF32(v))
    : maybeTrackOwnedTensor(ownedRecorderInputTensors, v, await ensureF32(v, recorder));
  const sTensor = !recorder
    ? maybeTrackOwnedTensor(ownedInputTensors, softmax, await ensureF32(softmax))
    : maybeTrackOwnedTensor(ownedRecorderInputTensors, softmax, await ensureF32(softmax, recorder));
  const dTensor = !recorder
    ? maybeTrackOwnedTensor(ownedInputTensors, gradOutput, await ensureF32(gradOutput))
    : maybeTrackOwnedTensor(ownedRecorderInputTensors, gradOutput, await ensureF32(gradOutput, recorder));

  const headElements = seqLen * headDim;
  const headBytes = headElements * dtypeBytes(qTensor.dtype);
  const softmaxBytes = seqLen * seqLen * dtypeBytes(sTensor.dtype);

  const totalBytes = numHeads * headBytes;
  const gradQBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_q');
  const gradKBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_k');
  const gradVBuf = acquireBuffer(totalBytes, undefined, 'attn_grad_v');
  let completed = false;

  try {
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
        let sTransposed = null;
        let dV = null;
        let vTransposed = null;
        let dS = null;
        let dQK = null;
        let dQ = null;
        let dQKTransposed = null;
        let dK = null;

        try {
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

          sTransposed = await runTranspose(sHead, seqLen, seqLen);
          dV = await runMatmul(sTransposed, dHead.buffer, seqLen, headDim, seqLen, {
            transposeB: false,
            bDtype: 'f32',
          });

          vTransposed = await runTranspose(vHead, seqLen, headDim);
          dS = await runMatmul(dHead, vTransposed.buffer, seqLen, seqLen, headDim, {
            transposeB: false,
            bDtype: 'f32',
          });
          dQK = causal
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

          dQ = await runMatmul(dQK, kHead.buffer, seqLen, headDim, seqLen, {
            transposeB: false,
            alpha: scale,
            bDtype: 'f32',
          });
          dQKTransposed = await runTranspose(dQK, seqLen, seqLen);
          dK = await runMatmul(dQKTransposed, qHead.buffer, seqLen, headDim, seqLen, {
            transposeB: false,
            alpha: scale,
            bDtype: 'f32',
          });

          const copyEncoder = getDevice().createCommandEncoder();
          copyEncoder.copyBufferToBuffer(dQ.buffer, 0, gradQBuf, qOffset, headBytes);
          copyEncoder.copyBufferToBuffer(dK.buffer, 0, gradKBuf, kOffset, headBytes);
          copyEncoder.copyBufferToBuffer(dV.buffer, 0, gradVBuf, vOffset, headBytes);
          getDevice().queue.submit([copyEncoder.finish()]);
          await getDevice().queue.onSubmittedWorkDone();
        } finally {
          releaseTensorBuffer(sTransposed);
          releaseTensorBuffer(dV);
          releaseTensorBuffer(vTransposed);
          releaseTensorBuffer(dS);
          releaseTensorBuffer(dQK);
          releaseTensorBuffer(dQ);
          releaseTensorBuffer(dQKTransposed);
          releaseTensorBuffer(dK);
          releaseBuffer(qHeadBuf);
          releaseBuffer(kHeadBuf);
          releaseBuffer(vHeadBuf);
          releaseBuffer(sHeadBuf);
          releaseBuffer(dHeadBuf);
        }
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
          headBytes,
          softmaxBytes
        );
        const headBuffers = [qHeadBuf, kHeadBuf, vHeadBuf, sHeadBuf, dHeadBuf];
        let sTransposed = null;
        let dV = null;
        let vTransposed = null;
        let dS = null;
        let dQK = null;
        let dQ = null;
        let dQKTransposed = null;
        let dK = null;

        try {
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

          sTransposed = await recordTranspose(recorder, sHead, seqLen, seqLen);
          dV = await recordMatmul(recorder, sTransposed, dHead.buffer, seqLen, headDim, seqLen, {
            transposeB: false,
            bDtype: 'f32',
          });

          vTransposed = await recordTranspose(recorder, vHead, seqLen, headDim);
          dS = await recordMatmul(recorder, dHead, vTransposed.buffer, seqLen, seqLen, headDim, {
            transposeB: false,
            bDtype: 'f32',
          });
          dQK = causal
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

          dQ = await recordMatmul(recorder, dQK, kHead.buffer, seqLen, headDim, seqLen, {
            transposeB: false,
            alpha: scale,
            bDtype: 'f32',
          });
          dQKTransposed = await recordTranspose(recorder, dQK, seqLen, seqLen);
          dK = await recordMatmul(recorder, dQKTransposed, qHead.buffer, seqLen, headDim, seqLen, {
            transposeB: false,
            alpha: scale,
            bDtype: 'f32',
          });

          encoder.copyBufferToBuffer(dQ.buffer, 0, gradQBuf, qOffset, headBytes);
          encoder.copyBufferToBuffer(dK.buffer, 0, gradKBuf, kOffset, headBytes);
          encoder.copyBufferToBuffer(dV.buffer, 0, gradVBuf, vOffset, headBytes);
        } catch (error) {
          releaseTensorBuffer(sTransposed);
          releaseTensorBuffer(dV);
          releaseTensorBuffer(vTransposed);
          releaseTensorBuffer(dS);
          releaseTensorBuffer(dQK);
          releaseTensorBuffer(dQ);
          releaseTensorBuffer(dQKTransposed);
          releaseTensorBuffer(dK);
          releaseBuffer(qHeadBuf);
          releaseBuffer(kHeadBuf);
          releaseBuffer(vHeadBuf);
          releaseBuffer(sHeadBuf);
          releaseBuffer(dHeadBuf);
          throw error;
        }

        for (const buffer of headBuffers) {
          recorder.trackTemporaryBuffer(buffer);
        }
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
    if (recorder) {
      for (const tensor of ownedRecorderInputTensors) {
        trackTensorBuffer(recorder, tensor);
      }
    }
    completed = true;
    return {
      gradQ: createTensor(gradQBuf, 'f32', [...q.shape], 'attn_grad_q'),
      gradK: createTensor(gradKBuf, 'f32', [...k.shape], 'attn_grad_k'),
      gradV: createTensor(gradVBuf, 'f32', [...v.shape], 'attn_grad_v'),
    };
  } finally {
    if (!completed) {
      releaseBuffer(gradQBuf);
      releaseBuffer(gradKBuf);
      releaseBuffer(gradVBuf);
    }
    if (!recorder) {
      for (const tensor of ownedInputTensors) {
        releaseTensorBuffer(tensor);
      }
    } else {
      for (const tensor of ownedRecorderInputTensors) {
        releaseTensorBuffer(tensor);
      }
    }
  }
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
  return runAttentionBackwardCore(q, k, v, softmax, gradOutput, options);
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
