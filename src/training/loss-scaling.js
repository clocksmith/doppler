
import { readBuffer } from '../memory/buffer-pool.js';
import { runScale } from '../gpu/kernels/index.js';
import { f16ToF32Array } from '../inference/kv-cache/types.js';

function toFloat32(buffer, dtype) {
  if (dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(buffer));
  }
  return new Float32Array(buffer);
}

export async function detectOverflow(grads) {
  for (const grad of grads.values()) {
    const data = toFloat32(await readBuffer(grad.buffer), grad.dtype);
    for (let i = 0; i < data.length; i += 1) {
      const value = data[i];
      if (!Number.isFinite(value)) {
        return true;
      }
    }
  }
  return false;
}

export class DynamicLossScaler {
  constructor(config) {
    this.enabled = Boolean(config?.enabled);
    this.scale = config?.initialScale ?? 1024;
    this.minScale = config?.minScale ?? 1;
    this.maxScale = config?.maxScale ?? 65536;
    this.scaleFactor = config?.scaleFactor ?? 2;
    this.backoffFactor = config?.backoffFactor ?? 0.5;
    this.growthInterval = config?.growthInterval ?? 2000;
    this.overflowCheck = config?.overflowCheck ?? true;
    this._growthSteps = 0;
  }

  shouldScale() {
    return this.enabled && this.scale !== 1;
  }

  async scaleLoss(loss) {
    if (!this.shouldScale()) {
      return loss;
    }
    return runScale(loss, this.scale, { inplace: true });
  }

  async unscaleGradients(grads) {
    if (!this.shouldScale()) {
      return grads;
    }
    const invScale = 1 / this.scale;
    const unscaled = new Map();
    for (const [param, grad] of grads.entries()) {
      const scaled = await runScale(grad, invScale, { inplace: true });
      unscaled.set(param, scaled);
    }
    return unscaled;
  }

  update(hasOverflow) {
    if (!this.enabled) {
      return;
    }
    if (hasOverflow) {
      const next = Math.max(this.minScale, this.scale * this.backoffFactor);
      this.scale = next;
      this._growthSteps = 0;
      return;
    }
    this._growthSteps += 1;
    if (this._growthSteps >= this.growthInterval) {
      const next = Math.min(this.maxScale, this.scale * this.scaleFactor);
      this.scale = next;
      this._growthSteps = 0;
    }
  }
}
