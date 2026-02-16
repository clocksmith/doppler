import { getDevice } from '../../../../gpu/device.js';
import { createCommandRecorder } from '../../../../gpu/command-recorder.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { acquireBuffer, readBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import {
  recordGeLU,
  recordMatmul,
  recordResidualAdd,
  recordScale,
  recordMatmulBackward,
  recordGeluBackward,
  recordAdam,
} from '../../../../gpu/kernels/index.js';
import { mlpForwardBatch } from './mlp.js';

function uploadF32Tensor(device, data, shape, label) {
  const byteLength = data.byteLength;
  const alignedSize = Math.ceil(byteLength / 4) * 4;
  const buffer = acquireBuffer(alignedSize, undefined, label);
  if (alignedSize === byteLength) {
    device.queue.writeBuffer(buffer, 0, data);
  } else {
    const padded = new Float32Array(alignedSize / 4);
    padded.set(data);
    device.queue.writeBuffer(buffer, 0, padded);
  }
  return createTensor(buffer, 'f32', shape, label);
}

function zeroF32Buffer(device, buffer, count) {
  const zeros = new Float32Array(count);
  device.queue.writeBuffer(buffer, 0, zeros);
}

export function createMlpTrainer(inputSize, hiddenSize, config = {}) {
  const device = getDevice();
  if (!device) {
    throw new Error('createMlpTrainer requires a GPU device.');
  }
  const safeInput = Math.max(1, Math.floor(inputSize));
  const safeHidden = Math.max(1, Math.floor(hiddenSize));

  const lr = Number.isFinite(config.lr) ? config.lr : 0.001;
  const beta1 = Number.isFinite(config.beta1) ? config.beta1 : 0.9;
  const beta2 = Number.isFinite(config.beta2) ? config.beta2 : 0.999;
  const eps = Number.isFinite(config.eps) ? config.eps : 1e-8;

  const w1Count = safeInput * safeHidden;
  const w2Count = safeHidden;

  const w1Moment1 = acquireBuffer(w1Count * 4, undefined, 'vliw_mlp_adam_m_w1');
  const w1Moment2 = acquireBuffer(w1Count * 4, undefined, 'vliw_mlp_adam_v_w1');
  const w2Moment1 = acquireBuffer(w2Count * 4, undefined, 'vliw_mlp_adam_m_w2');
  const w2Moment2 = acquireBuffer(w2Count * 4, undefined, 'vliw_mlp_adam_v_w2');

  zeroF32Buffer(device, w1Moment1, w1Count);
  zeroF32Buffer(device, w1Moment2, w1Count);
  zeroF32Buffer(device, w2Moment1, w2Count);
  zeroF32Buffer(device, w2Moment2, w2Count);

  return {
    inputSize: safeInput,
    hiddenSize: safeHidden,
    opt: {
      lr,
      beta1,
      beta2,
      eps,
      step: 1,
    },
    moments: {
      w1: {
        m: createTensor(w1Moment1, 'f32', [w1Count], 'vliw_mlp_m_w1'),
        v: createTensor(w1Moment2, 'f32', [w1Count], 'vliw_mlp_v_w1'),
      },
      w2: {
        m: createTensor(w2Moment1, 'f32', [w2Count], 'vliw_mlp_m_w2'),
        v: createTensor(w2Moment2, 'f32', [w2Count], 'vliw_mlp_v_w2'),
      },
    },
  };
}

export async function mlpTrainDistillStep(trainer, studentMlp, featureBatch, teacherMlp) {
  if (!trainer || !studentMlp || !featureBatch || !teacherMlp) {
    return studentMlp;
  }
  const device = getDevice();
  if (!device) {
    return studentMlp;
  }

  const inputSize = trainer.inputSize;
  const hiddenSize = trainer.hiddenSize;
  const rows = Math.floor(featureBatch.length / inputSize);
  if (rows <= 0) {
    return studentMlp;
  }

  const targets = mlpForwardBatch(teacherMlp, featureBatch, rows);
  const featuresTensor = uploadF32Tensor(device, featureBatch, [rows, inputSize], 'vliw_mlp_features');
  const targetTensor = uploadF32Tensor(device, targets, [rows, 1], 'vliw_mlp_targets');
  const w1Tensor = uploadF32Tensor(device, studentMlp.w1, [inputSize, hiddenSize], 'vliw_mlp_w1');
  const w2Tensor = uploadF32Tensor(device, studentMlp.w2, [hiddenSize, 1], 'vliw_mlp_w2');

  const buffersToRelease = new Set([featuresTensor.buffer, targetTensor.buffer, w1Tensor.buffer, w2Tensor.buffer]);
  let recorder = null;
  try {
    recorder = createCommandRecorder('vliw_mlp_train');

    const hidden = await recordMatmul(recorder, featuresTensor, w1Tensor, rows, hiddenSize, inputSize, {
      transposeB: false,
      role: 'vliw_mlp_w1',
    });
    recorder.trackTemporaryBuffer(hidden.buffer);
    const activated = await recordGeLU(recorder, hidden, { size: rows * hiddenSize });
    recorder.trackTemporaryBuffer(activated.buffer);
    const out = await recordMatmul(recorder, activated, w2Tensor, rows, 1, hiddenSize, {
      transposeB: false,
      role: 'vliw_mlp_w2',
    });
    recorder.trackTemporaryBuffer(out.buffer);

    const negTargets = await recordScale(recorder, targetTensor, -1.0, { inplace: true, count: rows });
    const diff = await recordResidualAdd(recorder, out, negTargets, rows);
    recorder.trackTemporaryBuffer(diff.buffer);
    const gradScale = rows > 0 ? (2.0 / rows) : 0.0;
    const gradOut = await recordScale(recorder, diff, gradScale, { inplace: true, count: rows });

    const dActivatedResult = await recordMatmulBackward(
      recorder,
      activated,
      w2Tensor,
      gradOut,
      { M: rows, N: 1, K: hiddenSize, transposeB: false }
    );
    if (!dActivatedResult.gradInput || !dActivatedResult.gradWeight) {
      throw new Error('mlpTrainDistillStep expected gradients for w2.');
    }
    recorder.trackTemporaryBuffer(dActivatedResult.gradInput.buffer);
    recorder.trackTemporaryBuffer(dActivatedResult.gradWeight.buffer);

    const dHidden = await recordGeluBackward(recorder, hidden, dActivatedResult.gradInput, {
      count: rows * hiddenSize,
    });
    recorder.trackTemporaryBuffer(dHidden.buffer);

    const dW1Result = await recordMatmulBackward(
      recorder,
      featuresTensor,
      w1Tensor,
      dHidden,
      { M: rows, N: hiddenSize, K: inputSize, transposeB: false, computeGradInput: false, computeGradWeight: true }
    );
    if (!dW1Result.gradWeight) {
      throw new Error('mlpTrainDistillStep expected gradients for w1.');
    }
    recorder.trackTemporaryBuffer(dW1Result.gradWeight.buffer);

    await recordAdam(recorder, w1Tensor, dW1Result.gradWeight, trainer.moments.w1.m, trainer.moments.w1.v, {
      count: inputSize * hiddenSize,
      step: trainer.opt.step,
      lr: trainer.opt.lr,
      beta1: trainer.opt.beta1,
      beta2: trainer.opt.beta2,
      eps: trainer.opt.eps,
    });

    await recordAdam(recorder, w2Tensor, dActivatedResult.gradWeight, trainer.moments.w2.m, trainer.moments.w2.v, {
      count: hiddenSize,
      step: trainer.opt.step,
      lr: trainer.opt.lr,
      beta1: trainer.opt.beta1,
      beta2: trainer.opt.beta2,
      eps: trainer.opt.eps,
    });

    await recorder.submitAndWait();
    trainer.opt.step += 1;

    const next = {
      inputSize,
      hiddenSize,
      w1: new Float32Array(studentMlp.w1.length),
      w2: new Float32Array(studentMlp.w2.length),
    };
    const w1Readback = await readBuffer(w1Tensor.buffer, studentMlp.w1.length * 4);
    next.w1.set(new Float32Array(w1Readback));

    const w2Readback = await readBuffer(w2Tensor.buffer, studentMlp.w2.length * 4);
    next.w2.set(new Float32Array(w2Readback));

    releaseBuffer(w1Tensor.buffer);
    releaseBuffer(w2Tensor.buffer);
    buffersToRelease.delete(w1Tensor.buffer);
    buffersToRelease.delete(w2Tensor.buffer);

    return next;
  } catch (error) {
    recorder?.abort();
    if (buffersToRelease.has(w1Tensor.buffer)) {
      releaseBuffer(w1Tensor.buffer);
      buffersToRelease.delete(w1Tensor.buffer);
    }
    if (buffersToRelease.has(w2Tensor.buffer)) {
      releaseBuffer(w2Tensor.buffer);
      buffersToRelease.delete(w2Tensor.buffer);
    }
    throw error;
  } finally {
    for (const buffer of buffersToRelease) {
      releaseBuffer(buffer);
    }
  }
}

export function disposeMlpTrainer(trainer) {
  if (!trainer) return;
  releaseBuffer(trainer.moments.w1.m.buffer);
  releaseBuffer(trainer.moments.w1.v.buffer);
  releaseBuffer(trainer.moments.w2.m.buffer);
  releaseBuffer(trainer.moments.w2.v.buffer);
}
