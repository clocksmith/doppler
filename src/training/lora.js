import { acquireBuffer, releaseBuffer, BufferUsage } from '../gpu/buffer-pool.js';
import { createTensor, tensorBytes } from '../gpu/tensor.js';
import { getTrainingConfig } from '../config/training-defaults.js';
import { runMatmul, runScale } from '../gpu/kernels/index.js';
import { OpType } from './autograd.js';

export class LoraAdapter {
  constructor(config) {
    const { inDim, outDim, rank, alpha } = config;
    const { loraParams: dtype } = getTrainingConfig().training.precision;

    const aBytes = tensorBytes([inDim, rank], dtype);
    const bBytes = tensorBytes([rank, outDim], dtype);

    this.A = createTensor(
      acquireBuffer(aBytes, BufferUsage.STORAGE, 'lora_A'),
      dtype,
      [inDim, rank],
      'lora_A'
    );
    this.B = createTensor(
      acquireBuffer(bBytes, BufferUsage.STORAGE, 'lora_B'),
      dtype,
      [rank, outDim],
      'lora_B'
    );
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
