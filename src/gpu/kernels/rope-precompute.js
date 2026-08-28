import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

const WORKGROUP_SIZE = 256;

export function planRoPEPrecomputeDispatch(device, count) {
  if (!Number.isSafeInteger(count) || count <= 0) {
    throw new Error(`[RoPE] frequency element count must be a positive safe integer; got "${count}".`);
  }
  const declaredLimit = device?.limits?.maxComputeWorkgroupsPerDimension;
  const maxPerDimension = Number.isFinite(declaredLimit) && declaredLimit > 0
    ? Math.floor(declaredLimit)
    : 65535;
  const totalWorkgroups = Math.ceil(count / WORKGROUP_SIZE);
  const y = Math.ceil(totalWorkgroups / maxPerDimension);
  if (y > maxPerDimension) {
    throw new Error(
      `[RoPE] frequency precompute requires ${totalWorkgroups} workgroups, exceeding ` +
      `the device's ${maxPerDimension} x ${maxPerDimension} dispatch capacity.`
    );
  }
  const x = Math.ceil(totalWorkgroups / y);
  return {
    dispatchStride: x * WORKGROUP_SIZE,
    workgroups: [x, y, 1],
  };
}

export function buildRoPEPrecomputeAuxiliaryData(options, halfDim) {
  if (options.positionPlan != null || options.mropeSection != null) {
    if (options.positionPlan == null || options.mropeSection == null) {
      throw new Error('[RoPE] positionPlan and mropeSection must be provided together.');
    }
    const sections = Array.from(options.mropeSection, (value) => Number(value));
    if (
      sections.length !== 3
      || sections.some((value) => !Number.isSafeInteger(value) || value <= 0)
      || sections.reduce((sum, value) => sum + value, 0) !== halfDim
    ) {
      throw new Error(`[RoPE] mropeSection must contain three positive integers summing to ${halfDim}.`);
    }
    const axes = [
      options.positionPlan.temporal,
      options.positionPlan.height,
      options.positionPlan.width,
    ];
    if (axes.some((axis) => !(axis instanceof Int32Array) || axis.length !== options.maxSeqLen)) {
      throw new Error(`[RoPE] MRoPE position axes must be Int32Array(${options.maxSeqLen}).`);
    }
    if (axes.some((axis) => axis.some((value) => value < 0))) {
      throw new Error('[RoPE] MRoPE position axes must contain non-negative integers.');
    }
    const factors = new Uint32Array(options.maxSeqLen * axes.length);
    factors.set(axes[0], 0);
    factors.set(axes[1], options.maxSeqLen);
    factors.set(axes[2], options.maxSeqLen * 2);
    return {
      code: 3,
      factors,
      yarn: null,
      originalMaxPosition: 1,
      mropeSection: sections,
    };
  }
  const type = options.scalingType ?? null;
  if (type == null || type === 'linear') {
    return {
      code: 0,
      factors: new Float32Array(1),
      yarn: null,
      originalMaxPosition: 1,
      mropeSection: [0, 0, 0],
    };
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
      mropeSection: [0, 0, 0],
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
    return {
      code: 2,
      factors,
      yarn: null,
      originalMaxPosition: original,
      mropeSection: [0, 0, 0],
    };
  }
  throw new Error(`[RoPE] unsupported scaling type "${type}".`);
}

export async function runRoPEPrecompute(options) {
  const device = getDevice();
  if (!device) throw new Error('[RoPE] GPU device is required for frequency precomputation.');
  const halfDim = options.rotaryDim / 2;
  const scaling = buildRoPEPrecomputeAuxiliaryData(options, halfDim);
  const count = options.maxSeqLen * halfDim;
  const dispatchPlan = planRoPEPrecomputeDispatch(device, count);
  const ropeData = acquireBuffer(
    Math.max(4, scaling.factors.byteLength),
    undefined,
    'rope_precompute_data'
  );
  const cos = acquireBuffer(count * Float32Array.BYTES_PER_ELEMENT, undefined, 'rope_cos');
  const sin = acquireBuffer(count * Float32Array.BYTES_PER_ELEMENT, undefined, 'rope_sin');
  device.queue.writeBuffer(ropeData, 0, scaling.factors);
  try {
    await unifiedKernelWrapper(
      'rope_precompute',
      null,
      'f32',
      [ropeData, cos, sin],
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
        dispatch_stride: dispatchPlan.dispatchStride,
        mrope_section_t: scaling.mropeSection[0],
        mrope_section_h: scaling.mropeSection[1],
        mrope_section_w: scaling.mropeSection[2],
      },
      dispatchPlan.workgroups
    );
    return { cos, sin };
  } catch (error) {
    releaseBuffer(cos);
    releaseBuffer(sin);
    throw error;
  } finally {
    releaseBuffer(ropeData);
  }
}
