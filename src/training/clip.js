
import { readBuffer } from '../gpu/buffer-pool.js';
import { runScale } from '../gpu/kernels/index.js';
import { f16ToF32Array } from '../inference/kv-cache/types.js';

async function readGradData(grad) {
  const buffer = await readBuffer(grad.buffer);
  if (grad.dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(buffer));
  }
  return new Float32Array(buffer);
}

export async function clipGradients(grads, config) {
  const maxNorm = config?.training?.gradient?.maxNorm;
  if (!maxNorm || maxNorm <= 0) {
    return grads;
  }

  let sumSq = 0;
  for (const grad of grads.values()) {
    const data = await readGradData(grad);
    for (let i = 0; i < data.length; i += 1) {
      const value = data[i];
      sumSq += value * value;
    }
  }

  const globalNorm = Math.sqrt(sumSq);
  if (!globalNorm || globalNorm <= maxNorm) {
    return grads;
  }

  const scale = maxNorm / (globalNorm + 1e-6);
  const clipped = new Map();
  for (const [param, grad] of grads.entries()) {
    const scaled = await runScale(grad, scale, { inplace: true });
    clipped.set(param, scaled);
  }

  return clipped;
}
