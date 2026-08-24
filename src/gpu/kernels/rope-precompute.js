import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

const WORKGROUP_SIZE = 256;

function resolveScaling(options, halfDim) {
  const type = options.scalingType ?? null;
  if (type == null || type === 'linear') {
    return { code: 0, factors: new Float32Array(1), yarn: null, originalMaxPosition: 1 };
  }
  if (type === 'yarn') {
    const scaling = options.scaling;
    for (const field of ['factor', 'beta_fast', 'beta_slow', 'original_max_position_embeddings']) {
      if (!Number.isFinite(Number(scaling?.[field]))) {
        throw new Error(`[RoPE] yarn scaling requires finite ${field}.`);
      }
    }
    return {
      code: 1,
      factors: new Float32Array(1),
      yarn: scaling,
      originalMaxPosition: Number(scaling.original_max_position_embeddings),
    };
  }
  if (type === 'longrope') {
    const scaling = options.scaling;
    const original = Number(scaling?.original_max_position_embeddings);
    const selected = options.maxSeqLen > original ? scaling?.long_factor : scaling?.short_factor;
    if (!Array.isArray(selected) || selected.length !== halfDim) {
      throw new Error(`[RoPE] longrope factors must contain ${halfDim} values.`);
    }
    const factors = Float32Array.from(selected, (value) => Number(value));
    if (!Number.isFinite(original) || original <= 1 || factors.some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new Error('[RoPE] longrope factors and original context must be positive and finite.');
    }
    return { code: 2, factors, yarn: null, originalMaxPosition: original };
  }
  throw new Error(`[RoPE] unsupported scaling type "${type}".`);
}

export async function runRoPEPrecompute(options) {
  const device = getDevice();
  if (!device) throw new Error('[RoPE] GPU device is required for frequency precomputation.');
  const halfDim = options.rotaryDim / 2;
  const scaling = resolveScaling(options, halfDim);
  const count = options.maxSeqLen * halfDim;
  const factors = acquireBuffer(
    Math.max(4, scaling.factors.byteLength),
    undefined,
    'rope_precompute_factors'
  );
  const cos = acquireBuffer(count * Float32Array.BYTES_PER_ELEMENT, undefined, 'rope_cos');
  const sin = acquireBuffer(count * Float32Array.BYTES_PER_ELEMENT, undefined, 'rope_sin');
  device.queue.writeBuffer(factors, 0, scaling.factors);
  try {
    await unifiedKernelWrapper(
      'rope_precompute',
      null,
      'f32',
      [factors, cos, sin],
      {
        max_seq_len: options.maxSeqLen,
        rotary_dim: options.rotaryDim,
        frequency_base_dim: options.frequencyBaseDim,
        scaling_type: scaling.code,
        theta: options.theta,
        rope_scale: options.ropeScale,
        yarn_factor: Number(scaling.yarn?.factor ?? 1),
        yarn_beta_fast: Number(scaling.yarn?.beta_fast ?? 1),
        yarn_beta_slow: Number(scaling.yarn?.beta_slow ?? 1),
        original_max_position: scaling.originalMaxPosition,
      },
      Math.ceil(count / WORKGROUP_SIZE)
    );
    return { cos, sin };
  } catch (error) {
    releaseBuffer(cos);
    releaseBuffer(sin);
    throw error;
  } finally {
    releaseBuffer(factors);
  }
}
