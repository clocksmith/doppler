import { getLayerSteps, getKernelPathStepConfig } from '../../../../config/kernel-path-loader.js';
import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer } from '../../../../memory/buffer-pool.js';
import { groupedPrefillGeometry, executeGroupedSort, executeGroupedProjection } from '../../../../gpu/kernels/moe-grouped.js';
import { runSiLURowSplit, recordSiLURowSplit } from '../../../../gpu/kernels/silu.js';
import { runRouteScatter, recordRouteScatter } from '../../../../gpu/kernels/moe-route.js';
import { releaseOrTrack } from '../ops.js';

/** @type {typeof import('./grouped-prefill.js').resolveGroupedPrefill} */
export function resolveGroupedPrefill(options) {
  if (options.kernelPath == null) return false;
  if (!['prefill', 'decode'].includes(options.phase) || !Number.isSafeInteger(options.layerIdx) || options.layerIdx < 0) {
    throw new Error('[MoE grouped prefill] Explicit phase and layer required.');
  }
  const steps = getLayerSteps(options.kernelPath, options.layerIdx, options.phase);
  const operations = ['moe_grouped_sort', 'moe_grouped_gu', 'moe_grouped_down'];
  const declared = steps.filter(step => operations.includes(step.op));
  if (!declared.length) return false;
  if (options.phase !== 'prefill') throw new Error('[MoE grouped prefill] Candidate cannot replace decode.');
  for (const [index, operation] of operations.entries()) {
    const step = declared[index];
    if (declared.length !== 3 || step?.op !== operation || !getKernelPathStepConfig(operation, step, options.phase)
      || (step.constants && Object.keys(step.constants).length)) {
      throw new Error('[MoE grouped prefill] Declare exactly sort, GU, down in order with registered kernels and model-bound geometry.');
    }
  }
  if (steps.some(step => step.op === 'moe_shared_slot_gu' || step.op === 'moe_shared_slot_down')) {
    throw new Error('[MoE grouped prefill] Routed-only grouping cannot combine with shared-slot execution in the same phase.');
  }
  return true;
}

/** @type {typeof import('./grouped-prefill.js').executeGroupedPrefill} */
export async function executeGroupedPrefill(recorder, options) {
  const { weights, inputTensor, numTokens, topK, hiddenSize, intermediateSize, swigluLimit } = options;
  if (weights.gateUp || !weights.gate || !weights.up) {
    throw new Error('[MoE grouped prefill] Separate native packed gate/up banks required; no implicit repacking.');
  }
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('[MoE grouped prefill] Active device required.');
  const geometry = { numTokens, numExperts: weights.numExperts, topK, hiddenSize, intermediateSize,
    kernelPath: options.kernelPath, phase: options.phase, layerIdx: options.layerIdx };
  const sizes = groupedPrefillGeometry(geometry, device.limits);
  const activate = recorder ? recordSiLURowSplit.bind(null, recorder) : runSiLURowSplit;
  const scatter = recorder ? recordRouteScatter.bind(null, recorder) : runRouteScatter;
  /** @type {GPUBuffer[]} */
  const owned = [];
  /** @param {GPUBuffer} buffer */
  const keep = buffer => { owned.push(buffer); return buffer; };
  try {
    const groups = keep(acquireBuffer(sizes.groupBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'moe_grouped_schedule'));
    const indirect = keep(acquireBuffer(24, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC, 'moe_grouped_indirect'));
    await executeGroupedSort(recorder, options.indicesBuffer, groups, indirect, geometry);
    const gateUp = await executeGroupedProjection(recorder, 'moe_grouped_gu', inputTensor,
      weights.gate, weights.up, groups, indirect, geometry);
    keep(gateUp.buffer);
    const activated = await activate(gateUp, { numTokens: sizes.numRoutes, dim: intermediateSize,
      activation: 'silu', swigluLimit });
    keep(activated.buffer);
    const down = await executeGroupedProjection(recorder, 'moe_grouped_down', activated,
      weights.down, null, groups, indirect, geometry);
    keep(down.buffer);
    return await scatter(down, options.weightsBuffer, { numTokens, hiddenSize, topK });
  } finally {
    for (const buffer of owned) releaseOrTrack(recorder, buffer);
  }
}
