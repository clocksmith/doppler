
import { trainStep } from './trainer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { AdamOptimizer } from './optimizer.js';
import { DynamicLossScaler, detectOverflow } from './loss-scaling.js';
import { readBuffer } from '../memory/buffer-pool.js';
import { f16ToF32Array } from '../inference/kv-cache/types.js';
import { DataLoader } from './dataloader.js';

function toFloat32(buffer, dtype) {
  if (dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(buffer));
  }
  return new Float32Array(buffer);
}

async function computeLossMean(loss) {
  const data = toFloat32(await readBuffer(loss.buffer), loss.dtype);
  if (!data.length) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) {
    sum += data[i];
  }
  return sum / data.length;
}

async function resolveBatches(dataset, batchSize, shuffle) {
  if (dataset && typeof dataset.batches === 'function') {
    return dataset.batches();
  }
  if (Array.isArray(dataset)) {
    const loader = new DataLoader(dataset, batchSize, shuffle);
    return loader.batches();
  }
  throw new Error('TrainingRunner requires dataset array or DataLoader');
}

export class TrainingRunner {
  constructor(config, options = {}) {
    this.config = config;
    this.optimizer = options.optimizer || new AdamOptimizer(config);
    this.lossFn = options.crossEntropyLoss || crossEntropyLoss;
    this.clipFn = options.clipGradients || clipGradients;
    this.lossScaler = options.lossScaler || new DynamicLossScaler(config.training.lossScaling);
    this.onStep = options.onStep || null;
    this.onEpoch = options.onEpoch || null;
  }

  async run(model, dataset, options = {}) {
    const {
      epochs = 1,
      batchSize = 1,
      shuffle = true,
      maxSteps = null,
      logEvery = 1,
      prepareBatch = null,
    } = options;

    let step = 0;
    const metrics = [];

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const batches = await resolveBatches(dataset, batchSize, shuffle);
      let batchIndex = 0;
      for await (const rawBatch of batches) {
        step += 1;
        batchIndex += 1;
        const batch = prepareBatch ? await prepareBatch(rawBatch) : rawBatch;
        const { loss } = await this._runStep(model, batch);
        const meanLoss = await computeLossMean(loss);
        const entry = { step, epoch, batch: batchIndex, loss: meanLoss };
        metrics.push(entry);

        if (this.onStep && (logEvery <= 0 || step % logEvery === 0)) {
          await this.onStep(entry);
        }

        if (maxSteps && step >= maxSteps) {
          if (this.onEpoch) {
            await this.onEpoch({ epoch, steps: batchIndex, loss: meanLoss });
          }
          return metrics;
        }
      }

      if (this.onEpoch) {
        const last = metrics[metrics.length - 1];
        await this.onEpoch({ epoch, steps: batchIndex, loss: last?.loss ?? 0 });
      }
    }

    return metrics;
  }

  async _runStep(model, batch) {
    const lossScale = this.lossScaler.shouldScale() ? this.lossScaler.scale : 1;
    const options = {
      crossEntropyLoss: this.lossFn,
      clipGradients: this.clipFn,
      optimizer: this.optimizer,
      lossScale,
      applyClip: false,
      applyOptimizer: false,
    };

    const result = await trainStep(model, batch, this.config, options);
    let grads = result.grads;

    if (this.lossScaler.enabled && this.lossScaler.overflowCheck) {
      const overflow = await detectOverflow(grads);
      this.lossScaler.update(overflow);
      if (overflow) {
        return { loss: result.loss };
      }
    } else if (this.lossScaler.enabled) {
      this.lossScaler.update(false);
    }

    grads = await this.clipFn(grads, this.config);
    await this.optimizer.step(model.loraParams(), grads, this.config);

    return { loss: result.loss };
  }
}

export async function runTraining(model, dataset, config, options = {}) {
  const runner = new TrainingRunner(config, options);
  return runner.run(model, dataset, options);
}
