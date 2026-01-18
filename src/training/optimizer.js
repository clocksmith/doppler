import { acquireBuffer, BufferUsage } from '../memory/buffer-pool.js';
import { createTensor, tensorBytes } from '../gpu/tensor.js';
import { runAdam } from '../gpu/kernels/backward/adam.js';

export class AdamOptimizer {
  constructor(config) {
    this.config = config;
    this.state = new Map();
    this.stepCount = 0;
  }

  getState(param) {
    let entry = this.state.get(param);
    if (!entry) {
      const bytes = tensorBytes(param.shape, param.dtype);
      const mBuf = acquireBuffer(bytes, BufferUsage.STORAGE, 'adam_m');
      const vBuf = acquireBuffer(bytes, BufferUsage.STORAGE, 'adam_v');
      entry = {
        m: createTensor(mBuf, param.dtype, [...param.shape], 'adam_m'),
        v: createTensor(vBuf, param.dtype, [...param.shape], 'adam_v'),
      };
      this.state.set(param, entry);
    }
    return entry;
  }

  async step(params, grads, trainingConfig) {
    const opt = trainingConfig.training.optimizer;
    this.stepCount += 1;

    for (const param of params) {
      const grad = grads.get(param);
      if (!grad) {
        continue;
      }

      const { m, v } = this.getState(param);
      await runAdam(param, grad, m, v, {
        step: this.stepCount,
        lr: opt.lr,
        beta1: opt.beta1,
        beta2: opt.beta2,
        eps: opt.eps,
      });
    }
  }
}
