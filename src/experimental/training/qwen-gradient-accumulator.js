import { runResidualAdd, runScale } from '../../gpu/kernels/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function elementCount(tensor) {
  if (!Array.isArray(tensor?.shape) || tensor.shape.length < 1) {
    throw new Error('Qwen gradient accumulator requires tensor shapes.');
  }
  return tensor.shape.reduce(
    (product, value) => product * positiveInteger(value, 'gradient dimension'),
    1
  );
}

function sameShape(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function releaseTensor(tensor) {
  if (tensor?.buffer) releaseBuffer(tensor.buffer);
}

function validateEntries(entries) {
  if (!Array.isArray(entries) || entries.length < 1) {
    throw new Error('Qwen gradient accumulator requires adapter entries.');
  }
  const names = new Set();
  const parameters = new Set();
  return entries.map((entry) => {
    const name = String(entry?.name || '');
    if (!name || names.has(name)) {
      throw new Error(`Qwen gradient accumulator duplicate or missing name "${name}".`);
    }
    if (!entry?.parameter?.buffer || !entry?.gradient?.buffer) {
      throw new Error(`Qwen gradient accumulator entry ${name} requires parameter and gradient tensors.`);
    }
    if (parameters.has(entry.parameter)) {
      throw new Error(`Qwen gradient accumulator parameter is duplicated at ${name}.`);
    }
    if (entry.gradient.dtype !== 'f32' || !sameShape(entry.parameter.shape, entry.gradient.shape)) {
      throw new Error(`Qwen gradient accumulator entry ${name} requires matching F32 gradients.`);
    }
    names.add(name);
    parameters.add(entry.parameter);
    return {
      name,
      parameter: entry.parameter,
      gradient: entry.gradient,
      elementCount: elementCount(entry.gradient),
    };
  });
}

export class QwenGradientAccumulator {
  constructor(options = {}) {
    this.accumSteps = positiveInteger(options.accumSteps, 'accumSteps');
    this.microstepCount = 0;
    this.entries = [];
  }

  get ready() {
    return this.microstepCount === this.accumSteps;
  }

  get parameterNames() {
    return this.entries.map((entry) => entry.name);
  }

  async accumulate(entries) {
    if (this.ready) {
      throw new Error('Qwen gradient accumulation window is full; apply or reset it before adding data.');
    }
    const normalized = validateEntries(entries);
    if (this.entries.length > 0) {
      if (normalized.length !== this.entries.length) {
        throw new Error('Qwen gradient accumulation parameter count changed within the window.');
      }
      for (let index = 0; index < normalized.length; index += 1) {
        const current = this.entries[index];
        const incoming = normalized[index];
        if (incoming.name !== current.name || incoming.parameter !== current.parameter
          || incoming.elementCount !== current.elementCount) {
          throw new Error(`Qwen gradient accumulation parameter identity changed at index ${index}.`);
        }
      }
    }

    const scaled = [];
    const combined = [];
    try {
      for (let index = 0; index < normalized.length; index += 1) {
        const incoming = normalized[index];
        const scaledGradient = await runScale(
          incoming.gradient,
          1 / this.accumSteps,
          { count: incoming.elementCount, inplace: false }
        );
        scaled.push(scaledGradient);
        if (this.entries.length > 0) {
          const sum = await runResidualAdd(
            this.entries[index].gradient,
            scaledGradient,
            incoming.elementCount
          );
          combined.push(sum);
        }
      }
    } catch (error) {
      for (const tensor of scaled) releaseTensor(tensor);
      for (const tensor of combined) releaseTensor(tensor);
      throw error;
    }

    if (this.entries.length === 0) {
      this.entries = normalized.map((entry, index) => ({
        name: entry.name,
        parameter: entry.parameter,
        gradient: scaled[index],
        elementCount: entry.elementCount,
      }));
    } else {
      for (const entry of this.entries) releaseTensor(entry.gradient);
      for (const tensor of scaled) releaseTensor(tensor);
      this.entries = normalized.map((entry, index) => ({
        name: entry.name,
        parameter: entry.parameter,
        gradient: combined[index],
        elementCount: entry.elementCount,
      }));
    }
    this.microstepCount += 1;
    return {
      microstepCount: this.microstepCount,
      accumSteps: this.accumSteps,
      ready: this.ready,
      parameterCount: this.entries.length,
    };
  }

  async step(optimizer, trainingConfig) {
    if (!this.ready) {
      throw new Error(
        `Qwen gradient accumulation window is incomplete: ${this.microstepCount}/${this.accumSteps}.`
      );
    }
    if (!optimizer || typeof optimizer.step !== 'function') {
      throw new Error('Qwen gradient accumulator requires an optimizer to apply the window.');
    }
    const parameters = this.entries.map((entry) => entry.parameter);
    const gradients = new Map(
      this.entries.map((entry) => [entry.parameter, entry.gradient])
    );
    const metrics = await optimizer.step(parameters, gradients, trainingConfig);
    this.reset();
    return metrics;
  }

  reset() {
    for (const entry of this.entries) releaseTensor(entry.gradient);
    this.entries = [];
    this.microstepCount = 0;
  }

  dispose() {
    this.reset();
  }
}
