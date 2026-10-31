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
    throw new Error(`[MoE norm router] ${label} must be a row-major F32 [${rows}, ${columns}] weight.`);
  }
  return buffer;
}

/**
 * @param {import('../tensor.js').Tensor} tensor
 * @param {number} hiddenSize
 * @param {string} label
 */
function f32Row(tensor, hiddenSize, label) {
  if (tensor.dtype !== 'f32' || tensor.precision != null || tensor.buffer.size < hiddenSize * 4) {
    throw new Error(`[MoE norm router] ${label} must be one scalar F32 row of ${hiddenSize} values.`);
  }
  return tensor.buffer;
}

/** @type {typeof import('./moe-norm-router.js').recordMoENormRouter} */
export async function recordMoENormRouter(recorder, options) {
  const { hiddenSize, numExperts } = options;
  if (!Number.isSafeInteger(hiddenSize) || hiddenSize < 4 || hiddenSize % 4 !== 0
    || !Number.isSafeInteger(numExperts) || numExperts < 1 || !(options.eps > 0)) {
    throw new Error('[MoE norm router] Positive hiddenSize (a multiple of 4), numExperts and eps are required.');
  }
  // The fused kernel reproduces rmsnorm/default and matmul/gemv_f32_subgroup only at
  // their registered geometry; a registry change to either must change both.
  const fused = getKernelConfig('moe_norm_router', 'f32');
  const norm = getKernelConfig('rmsnorm', 'default');
  const gemv = getKernelConfig('matmul', 'gemv_f32_subgroup');
  const { WORKGROUP_SIZE: workgroupSize, COLS_PER_WG: colsPerWorkgroup, NORM_THREADS: normThreads } = fused.wgslOverrides ?? {};
  if (typeof workgroupSize !== 'number' || typeof colsPerWorkgroup !== 'number' || typeof normThreads !== 'number'
    || normThreads !== norm.workgroupSize?.[0] || workgroupSize !== gemv.wgslOverrides?.WORKGROUP_SIZE
    || colsPerWorkgroup !== gemv.wgslOverrides?.COLS_PER_WG) {
    throw new Error('[MoE norm router] Registered geometry no longer matches rmsnorm/default and matmul/gemv_f32_subgroup.');
  }
  const device = recorder.device;
  if (hiddenSize * 4 + normThreads * 4 > device.limits.maxComputeWorkgroupStorageSize) {
    throw new Error('[MoE norm router] The normalized row exceeds workgroup storage.');
  }
  const outputs = [options.normedOutput, options.prenormOutput, options.logitsOutput, options.sharedGateOutput];
  const sizes = [hiddenSize * 4, hiddenSize * 4, numExperts * 4, 4];
  if (outputs.some((buffer, i) => !buffer || buffer.size < sizes[i]) || new Set(outputs).size !== outputs.length) {
    throw new Error('[MoE norm router] Four distinct outputs of the declared sizes are required.');
  }
  const normWeight = isWeightBuffer(options.normWeight) ? options.normWeight.buffer : options.normWeight;
  const columns = numExperts + 1;
  const workgroupsX = Math.ceil(columns / colsPerWorkgroup);
  await unifiedKernelWrapper('moe_norm_router', recorder, 'f32', [
    f32Row(options.input, hiddenSize, 'input'),
    f32Row(options.residual, hiddenSize, 'residual'),
    normWeight,
    f32RowWeight(options.routerWeight, numExperts, hiddenSize, 'router weight'),
    f32RowWeight(options.sharedRouterWeight, 1, hiddenSize, 'shared router weight'),
    ...outputs,
  ], { hidden_size: hiddenSize, num_experts: numExperts, eps: options.eps, workgroups_x: workgroupsX },
  [workgroupsX, 1, 1], {
    WORKGROUP_SIZE: workgroupSize,
    COLS_PER_WG: colsPerWorkgroup,
    NORM_THREADS: normThreads,
    HIDDEN_SIZE: hiddenSize,
    RMS_NORM_OFFSET: options.rmsNormWeightOffset,
    // The norm weight dtype resolves as rmsnorm.js resolves it.
    WEIGHT_IS_F16: resolveNormWeightDtype(options.normWeight, hiddenSize) === 'f16',
  });
}
