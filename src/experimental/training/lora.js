import { acquireBuffer, releaseBuffer, uploadData, BufferUsage } from '../../memory/buffer-pool.js';
import { createTensor, tensorBytes } from '../../gpu/tensor.js';
import { getTrainingConfig } from '../../config/training-defaults.js';
import { runMatmul, runScale } from '../../gpu/kernels/index.js';
import { OpType } from './autograd.js';
import { f32ToF16Array } from '../../inference/kv-cache/types.js';

function createDeterministicInit(length, scale) {
  const values = new Float32Array(length);
  let state = 1337;
  for (let index = 0; index < length; index += 1) {
    state = (state * 16807) % 2147483647;
    values[index] = ((state / 2147483647) - 0.5) * scale;
  }
  return values;
}

function uploadLoraInit(buffer, dtype, values) {
  if (dtype === 'f16') {
    uploadData(buffer, f32ToF16Array(values));
    return;
  }
  uploadData(buffer, values);
}

export class LoraAdapter {
  constructor(config) {
    const { inDim, outDim, rank, alpha } = config;
    const { loraParams: dtype } = getTrainingConfig().training.precision;

    const aBytes = tensorBytes([inDim, rank], dtype);
    const bBytes = tensorBytes([rank, outDim], dtype);

    let aBuffer = null;
    let bBuffer = null;
    try {
      aBuffer = acquireBuffer(aBytes, BufferUsage.STORAGE, 'lora_A');
      bBuffer = acquireBuffer(bBytes, BufferUsage.STORAGE, 'lora_B');
      this.A = createTensor(
        aBuffer,
        dtype,
        [inDim, rank],
        'lora_A'
      );
      this.B = createTensor(
        bBuffer,
        dtype,
        [rank, outDim],
        'lora_B'
      );
      const initScale = 0.02 / Math.max(1, Math.sqrt(inDim));
      uploadLoraInit(aBuffer, dtype, createDeterministicInit(inDim * rank, initScale));
      uploadLoraInit(bBuffer, dtype, new Float32Array(rank * outDim));
    } catch (error) {
      if (aBuffer) {
        releaseBuffer(aBuffer);
      }
      if (bBuffer) {
        releaseBuffer(bBuffer);
      }
      throw error;
    }
    this.alpha = alpha;
    this.rank = rank;
  }

  async forward(input, tape) {
    const [tokens] = input.shape;
    const down = await tape.record(
      OpType.MATMUL,
      (a, b) => runMatmul(a, b, tokens, this.rank, this.A.shape[0]),
      [input, this.A],
      { M: tokens, N: this.rank, K: this.A.shape[0] }
    );
    const up = await tape.record(
      OpType.MATMUL,
      (a, b) => runMatmul(a, b, tokens, this.B.shape[1], this.rank),
      [down, this.B],
      { M: tokens, N: this.B.shape[1], K: this.rank }
    );
    return tape.record(
      OpType.SCALE,
      (x) => runScale(x, this.alpha / this.rank),
      [up],
      { scale: this.alpha / this.rank }
    );
  }

  dispose() {
    releaseBuffer(this.A.buffer);
    releaseBuffer(this.B.buffer);
  }
}
