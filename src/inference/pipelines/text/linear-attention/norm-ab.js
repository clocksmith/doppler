import { getLayerSteps, getKernelPathStepConfig } from '../../../../config/kernel-path-loader.js';
import { recordLinearNormAB } from '../../../../gpu/kernels/linear-norm-ab.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { acquireBuffer } from '../../../../memory/buffer-pool.js';
import { isGpuBuffer, releaseOrTrackBuffer } from './plan.js';

/** @type {typeof import('./norm-ab.js').resolveLinearNormAB} */
export function resolveLinearNormAB(kernelPath, layerIdx, phase) {
  if (kernelPath == null) return null;
  const steps = getLayerSteps(kernelPath, layerIdx, phase).filter(step => step.op === 'linear_norm_ab');
  if (steps.length === 0) return null;
  const config = steps.length === 1 && phase === 'decode'
    ? getKernelPathStepConfig('linear_norm_ab', steps[0], phase) : null;
  if (!config || (steps[0].constants && Object.keys(steps[0].constants).length)
    || (config.variant !== 'f32' && config.variant !== 'f32_single_pass')) {
    throw new Error(`Layer ${layerIdx} linear_norm_ab requires one registered decode step with model-bound geometry.`);
  }
  return config.variant;
}

/** @type {typeof import('./norm-ab.js').recordDeclaredLinearNormAB} */
export async function recordDeclaredLinearNormAB(recorder, options) {
  const { layerIdx, inputTensor, layerWeights, hiddenSize, rows } = options;
  const variant = resolveLinearNormAB(options.kernelPath, layerIdx, options.phase);
  if (variant == null) return null;
  if (!recorder || options.numTokens !== 1 || options.precomputedInputNorm || !layerWeights.inputNorm
    || inputTensor.dtype !== 'f32' || inputTensor.precision != null || options.projectionDtype !== 'f32'
    || options.hasABLoRA) {
    if (options.precomputedInputNorm) releaseOrTrackBuffer(recorder, options.precomputedInputNorm.buffer);
    throw new Error(`linear_attention layer ${layerIdx} declares linear_norm_ab, which covers recorded `
      + 'single-token scalar F32 decode of its own input norm without a/b LoRA.');
  }
  const workspace = recorder.decodeWorkspace;
  /** @param {string} name @param {number} bytes */
  const output = (name, bytes) => workspace?.getBuffer(`linear_norm_ab:${layerIdx}:${name}`, bytes)
    ?? acquireBuffer(bytes, undefined, `linear_norm_ab_${name}`);
  const normed = output('normed', hiddenSize * 4);
  const a = output('a', rows * 4);
  const b = output('b', rows * 4);
  const normWeight = options.getNormWeightBuffer(layerWeights.inputNorm, `L${layerIdx}.linear_input_norm`);
  let recorded = false;
  try {
    await recordLinearNormAB(recorder, {
      variant, input: inputTensor, normWeight, rmsNormWeightOffset: options.rmsNormWeightOffset, eps: options.eps,
      aWeight: layerWeights.linearInProjA, bWeight: layerWeights.linearInProjB, hiddenSize, rows,
      normedOutput: normed, aOutput: a, bOutput: b,
    });
    recorded = true;
  } finally {
    if (!isGpuBuffer(layerWeights.inputNorm)) releaseOrTrackBuffer(recorder, normWeight);
    // Workspace outputs are persistent; tracking ignores them.
    if (!recorded) for (const buffer of [normed, a, b]) releaseOrTrackBuffer(recorder, buffer);
  }
  return {
    normed: createTensor(normed, 'f32', [1, hiddenSize], 'linear_input_norm'),
    a: createTensor(a, 'f32', [1, rows], 'linear_a'),
    b: createTensor(b, 'f32', [1, rows], 'linear_b'),
  };
}
