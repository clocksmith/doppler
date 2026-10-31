import { getKernelConfig } from './kernel-configs.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { isGpuBufferInstance, isWeightBuffer, getWeightDtype } from '../weight-buffer.js';
import { resolveNormWeightDtype } from './rmsnorm.js';

/**
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} weight
 * @param {number} rows
 * @param {number} columns
 * @param {string} label
 * @returns {GPUBuffer}
 */
function f32RowWeight(weight, rows, columns, label) {
  const buffer = isWeightBuffer(weight) ? weight.buffer : weight;
  const dtype = isWeightBuffer(weight) ? getWeightDtype(weight) : 'f32';
  const layout = isWeightBuffer(weight) ? weight.layout : 'row';
  if (!isGpuBufferInstance(buffer) || dtype !== 'f32' || layout !== 'row' || buffer.size < rows * columns * 4) {
    throw new Error(`[Linear norm a/b] ${label} must be a row-major F32 [${rows}, ${columns}] weight.`);
  }
  return buffer;
}

/** @type {typeof import('./linear-norm-ab.js').recordLinearNormAB} */
export async function recordLinearNormAB(recorder, options) {
  const { hiddenSize, rows } = options;
  if (!Number.isSafeInteger(hiddenSize) || hiddenSize < 4 || hiddenSize % 4 !== 0
    || !Number.isSafeInteger(rows) || rows < 1 || !(options.eps > 0)) {
    throw new Error('[Linear norm a/b] Positive hiddenSize (a multiple of 4), rows and eps are required.');
  }
  // Both variants reproduce rmsnorm/default's normalized row at its registered thread
  // count; f32 also reproduces matmul/gemv_f32_subgroup's columns (one subgroup per
  // column at any column count). A registry change to a reproduced kernel must change
  // these too.
  const variant = options.variant;
  if (variant !== 'f32' && variant !== 'f32_single_pass') {
    throw new Error('[Linear norm a/b] Explicit registered variant required (f32 or f32_single_pass).');
  }
  const fused = getKernelConfig('linear_norm_ab', variant);
  const norm = getKernelConfig('rmsnorm', 'default');
  const { WORKGROUP_SIZE: workgroupSize, COLS_PER_WG: colsPerWorkgroup, NORM_THREADS: normThreads, ROWS: rowsPerWorkgroup } = fused.wgslOverrides ?? {};
  const singlePass = variant === 'f32_single_pass';
  if (typeof normThreads !== 'number' || normThreads !== norm.workgroupSize?.[0] || (singlePass
    ? typeof rowsPerWorkgroup !== 'number' || rowsPerWorkgroup < 1 || rowsPerWorkgroup > 4
    : typeof workgroupSize !== 'number' || typeof colsPerWorkgroup !== 'number')) {
    throw new Error('[Linear norm a/b] Registered geometry no longer matches the reproduced kernels.');
  }
  const device = recorder.device;
  // The staged row, the norm partial sums and (single pass) four row dots per thread.
  if (hiddenSize * 4 + normThreads * (singlePass ? 20 : 4) > device.limits.maxComputeWorkgroupStorageSize) {
    throw new Error('[Linear norm a/b] The normalized row exceeds workgroup storage.');
  }
  const input = options.input;
  if (input.dtype !== 'f32' || input.precision != null || input.buffer.size < hiddenSize * 4) {
    throw new Error(`[Linear norm a/b] input must be one scalar F32 row of ${hiddenSize} values.`);
  }
  const outputs = [options.normedOutput, options.aOutput, options.bOutput];
  const sizes = [hiddenSize * 4, rows * 4, rows * 4];
  if (outputs.some((buffer, i) => !buffer || buffer.size < sizes[i]) || new Set([input.buffer, ...outputs]).size !== 4) {
    throw new Error('[Linear norm a/b] Three distinct outputs of the declared sizes are required.');
  }
  const normWeight = isWeightBuffer(options.normWeight) ? options.normWeight.buffer : options.normWeight;
  const workgroupsX = Math.ceil(2 * rows / /** @type {number} */ (singlePass ? rowsPerWorkgroup : colsPerWorkgroup));
  await unifiedKernelWrapper('linear_norm_ab', recorder, variant, [
    input.buffer,
    normWeight,
    f32RowWeight(options.aWeight, rows, hiddenSize, 'in_proj_a weight'),
    f32RowWeight(options.bWeight, rows, hiddenSize, 'in_proj_b weight'),
    ...outputs,
  ], { hidden_size: hiddenSize, rows, eps: options.eps, workgroups_x: workgroupsX },
  [workgroupsX, 1, 1], {
    ...(singlePass
      ? { ROWS: /** @type {number} */ (rowsPerWorkgroup) }
      : { WORKGROUP_SIZE: /** @type {number} */ (workgroupSize), COLS_PER_WG: /** @type {number} */ (colsPerWorkgroup) }),
    NORM_THREADS: normThreads,
    HIDDEN_SIZE: hiddenSize,
    RMS_NORM_OFFSET: options.rmsNormWeightOffset,
    // The norm weight dtype resolves as rmsnorm.js resolves it.
    WEIGHT_IS_F16: resolveNormWeightDtype(options.normWeight, hiddenSize) === 'f16',
  });
}
