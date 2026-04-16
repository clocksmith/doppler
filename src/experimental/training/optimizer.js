import { acquireBuffer, releaseBuffer, BufferUsage } from '../../memory/buffer-pool.js';
import { createTensor, tensorBytes } from '../../gpu/tensor.js';
import { runAdam } from '../../gpu/kernels/backward/adam.js';

function clampMin(value, minValue) {
  return value < minValue ? minValue : value;
}

function resolveEffectiveLr(opt, stepCount) {
  const baseLr = Number.isFinite(opt?.lr) ? opt.lr : 0;
  const scheduler = opt?.scheduler;
  if (!scheduler || scheduler.enabled === false) {
    return {
      effectiveLr: baseLr,
      schedulerIndex: Math.max(0, stepCount - 1),
      schedulerPhase: 'constant',
    };
  }

  const warmupSteps = Math.max(0, Math.floor(Number(scheduler.warmupSteps) || 0));
  const minLr = Number.isFinite(scheduler.minLr) ? scheduler.minLr : 0;
  if (warmupSteps > 0 && stepCount <= warmupSteps) {
    const warmupRatio = stepCount / warmupSteps;
    return {
      effectiveLr: clampMin(baseLr * warmupRatio, minLr),
      schedulerIndex: stepCount - 1,
      schedulerPhase: 'warmup',
    };
  }

  const schedulerType = typeof scheduler.type === 'string'
    ? scheduler.type
    : 'constant';
  if (schedulerType === 'step_decay') {
    const stepSize = Math.max(1, Math.floor(Number(scheduler.stepSize) || 1));
    const gamma = Number.isFinite(scheduler.gamma) ? scheduler.gamma : 1;
    const decayIndex = Math.floor(Math.max(0, stepCount - warmupSteps - 1) / stepSize);
    return {
      effectiveLr: clampMin(baseLr * (gamma ** decayIndex), minLr),
      schedulerIndex: decayIndex,
      schedulerPhase: 'decay',
    };
  }

  if (schedulerType === 'cosine') {
    const totalSteps = Math.max(1, Math.floor(Number(scheduler.totalSteps) || 1));
    const localStep = Math.max(0, stepCount - warmupSteps - 1);
    const progress = Math.min(1, localStep / totalSteps);
    const cosine = 0.5 * (1 + Math.cos(Math.PI * progress));
    return {
      effectiveLr: clampMin(baseLr * cosine, minLr),
      schedulerIndex: localStep,
      schedulerPhase: 'cosine',
    };
  }

  return {
    effectiveLr: baseLr,
    schedulerIndex: Math.max(0, stepCount - warmupSteps - 1),
    schedulerPhase: schedulerType || 'constant',
  };
}

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
      let mBuf = null;
      let vBuf = null;
      try {
        mBuf = acquireBuffer(bytes, BufferUsage.STORAGE, 'adam_m');
        vBuf = acquireBuffer(bytes, BufferUsage.STORAGE, 'adam_v');
        entry = {
          m: createTensor(mBuf, param.dtype, [...param.shape], 'adam_m'),
          v: createTensor(vBuf, param.dtype, [...param.shape], 'adam_v'),
        };
      } catch (error) {
        if (mBuf) {
          releaseBuffer(mBuf);
        }
        if (vBuf) {
          releaseBuffer(vBuf);
        }
        throw error;
      }
      this.state.set(param, entry);
    }
    return entry;
  }

  async step(params, grads, trainingConfig, _context = null) {
    const t0 = globalThis.performance.now();
    const opt = trainingConfig.training.optimizer;
    this.stepCount += 1;
    const lrResolved = resolveEffectiveLr(opt, this.stepCount);

    for (const param of params) {
      const grad = grads.get(param);
      if (!grad) {
        continue;
      }

      const { m, v } = this.getState(param);
      await runAdam(param, grad, m, v, {
        step: this.stepCount,
        lr: lrResolved.effectiveLr,
        beta1: opt.beta1,
        beta2: opt.beta2,
        eps: opt.eps,
      });
    }

    return {
      optimizer_ms: globalThis.performance.now() - t0,
      effective_lr: lrResolved.effectiveLr,
      scheduler_index: lrResolved.schedulerIndex,
      scheduler_phase: lrResolved.schedulerPhase,
    };
  }
}
