import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { Q4K_BLOCK_BYTES, q4kBlockCount } from '../../config/schema/index.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import {
  getBuffer,
  getLayout,
  getWeightDtype,
  isWeightBuffer,
  resolveWeightBufferMaterialization,
} from '../weight-buffer.js';
import {
  createPipeline,
  createUniformBufferWithView,
  getOrCreateBindGroupLayout,
} from './utils.js';
import { recordDispatch } from './dispatch.js';

const WORKGROUP_SIZE = 256;
const COLS_PER_WG = 64;
const THREADS_PER_COL = 4;
const MAX_COLS_PER_WG = 256;
const LM_HEAD_Q4K_FULL_BLOCK_FAST_PATH_VARIANTS = new Set(['phase1_q4k_f32a']);

function getLmHeadArgmaxBindGroupLayout(device) {
  return getOrCreateBindGroupLayout(
    'lm_head_argmax_bind_group_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
}

async function createLmHeadArgmaxPipeline(device, variant) {
  return createPipeline(
    'lm_head_argmax',
    variant,
    getLmHeadArgmaxBindGroupLayout(device),
    resolveLmHeadArgmaxPipelineConstants(variant)
  );
}

function resolveLmHeadArgmaxPipelineConstants(variant) {
  const tuning = getRuntimeConfig()?.inference?.session?.lmHeadArgmaxQ4K;
  const colsPerWorkgroup = tuning?.colsPerWorkgroup ?? COLS_PER_WG;
  const threadsPerCol = tuning?.threadsPerCol ?? THREADS_PER_COL;
  if (!Number.isInteger(colsPerWorkgroup) || colsPerWorkgroup <= 0 || colsPerWorkgroup > MAX_COLS_PER_WG) {
    throw new Error(
      `[LmHeadArgmax] lmHeadArgmaxQ4K.colsPerWorkgroup must be an integer in 1..${MAX_COLS_PER_WG}, ` +
      `got "${String(colsPerWorkgroup)}".`
    );
  }
  if (!Number.isInteger(threadsPerCol) || threadsPerCol <= 0) {
    throw new Error(
      `[LmHeadArgmax] lmHeadArgmaxQ4K.threadsPerCol must be a positive integer, got "${String(threadsPerCol)}".`
    );
  }
  if (colsPerWorkgroup * threadsPerCol !== WORKGROUP_SIZE) {
    throw new Error(
      `[LmHeadArgmax] lmHeadArgmaxQ4K requires colsPerWorkgroup * threadsPerCol == ${WORKGROUP_SIZE}; ` +
      `got ${colsPerWorkgroup} * ${threadsPerCol}.`
    );
  }
  const constants = {
    WORKGROUP_SIZE,
    COLS_PER_WG: colsPerWorkgroup,
    THREADS_PER_COL: threadsPerCol,
  };
  if (LM_HEAD_Q4K_FULL_BLOCK_FAST_PATH_VARIANTS.has(variant)) {
    constants.USE_FULL_BLOCK_FAST_PATH =
      tuning?.useFullBlockFastPath === true;
  }
  return constants;
}

function assertOutputBufferSize(outputBuffer, outputIndex) {
  const minBytes = Math.max(4, (outputIndex + 1) * 4);
  if (outputBuffer.size < minBytes) {
    throw new Error(
      `[LmHeadArgmax] outputBuffer too small for outputIndex=${outputIndex}: ` +
      `${outputBuffer.size} < ${minBytes}.`
    );
  }
}

function validateArgmaxOptions(options) {
  const vocabSize = options.vocabSize;
  const hiddenSize = options.hiddenSize;
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new Error(`[LmHeadArgmax] vocabSize must be a positive integer, got "${vocabSize}".`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`[LmHeadArgmax] hiddenSize must be a positive integer, got "${hiddenSize}".`);
  }
  if (options.outputIndex == null) {
    throw new Error('[LmHeadArgmax] outputIndex is required.');
  }
  if (options.logitSoftcap === undefined) {
    throw new Error('[LmHeadArgmax] logitSoftcap is required.');
  }
  if (options.padTokenId === undefined) {
    throw new Error('[LmHeadArgmax] padTokenId is required.');
  }
}

function assertWeightShape(resolved, layout, vocabSize, hiddenSize, dtype) {
  const shape = resolved.shape;
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error(`[LmHeadArgmax] LM head fusion requires 2D weights, got [${shape?.join?.(', ') ?? ''}].`);
  }
  const shapeVocab = layout === 'row' ? shape[0] : shape[1];
  const shapeHidden = layout === 'row' ? shape[1] : shape[0];
  if (shapeVocab < vocabSize || shapeHidden !== hiddenSize) {
    throw new Error(
      `[LmHeadArgmax] LM head shape mismatch: dtype=${dtype}, layout=${layout}, ` +
      `shape=[${shape.join(', ')}], vocab=${vocabSize}, hidden=${hiddenSize}.`
    );
  }
  return { shapeVocab, shapeHidden };
}

function resolveLmHeadWeightForDtype(lmHead, vocabSize, hiddenSize, expectedDtype) {
  if (!isWeightBuffer(lmHead)) {
    throw new Error('[LmHeadArgmax] LM head fusion requires a GPU-resident WeightBuffer.');
  }
  const resolved = resolveWeightBufferMaterialization(lmHead, expectedDtype);
  if (!isWeightBuffer(resolved)) {
    throw new Error('[LmHeadArgmax] LM head fusion requires a single GPU-resident weight buffer.');
  }
  const dtype = getWeightDtype(resolved);
  if (dtype !== expectedDtype) {
    throw new Error(`[LmHeadArgmax] LM head ${expectedDtype} fusion requires ${expectedDtype} weights, got "${dtype ?? 'unknown'}".`);
  }
  const layout = getLayout(resolved);
  if (expectedDtype === 'q4k' && layout !== 'row') {
    throw new Error(`[LmHeadArgmax] LM head q4k fusion requires row layout, got "${layout ?? 'unknown'}".`);
  }
  if (expectedDtype !== 'q4k' && layout !== 'row' && layout !== 'column') {
    throw new Error(
      `[LmHeadArgmax] LM head ${expectedDtype} fusion requires row or column layout, got "${layout ?? 'unknown'}".`
    );
  }
  const { shapeVocab } = assertWeightShape(resolved, layout, vocabSize, hiddenSize, expectedDtype);
  if (expectedDtype === 'q4k') {
    const minBytes = shapeVocab * q4kBlockCount(hiddenSize) * Q4K_BLOCK_BYTES;
    if (getBuffer(resolved).size < minBytes) {
      throw new Error(
        `[LmHeadArgmax] LM head q4k buffer too small: ${getBuffer(resolved).size} < ${minBytes}.`
      );
    }
  }
  const phase1Variant = selectRuleValue('kernels', 'lmHeadArgmax', 'phase1Variant', { weightDtype: expectedDtype });
  const phase2Variant = selectRuleValue('kernels', 'lmHeadArgmax', 'phase2Variant', { weightDtype: expectedDtype });
  return {
    buffer: getBuffer(resolved),
    dtype: expectedDtype,
    transposeB: layout === 'row',
    phase1Variant,
    phase2Variant,
  };
}

function resolveLmHeadWeight(lmHead, vocabSize, hiddenSize) {
  if (!isWeightBuffer(lmHead)) {
    throw new Error('[LmHeadArgmax] LM head fusion requires a GPU-resident WeightBuffer.');
  }
  const q4kPolicy = getRuntimeConfig()?.inference?.session?.lmHeadArgmaxQ4K;
  if (q4kPolicy != null) {
    const q4kResolved = resolveWeightBufferMaterialization(lmHead, 'q4k');
    if (isWeightBuffer(q4kResolved) && getWeightDtype(q4kResolved) === 'q4k') {
      return resolveLmHeadWeightForDtype(lmHead, vocabSize, hiddenSize, 'q4k');
    }
  }
  return resolveLmHeadWeightForDtype(lmHead, vocabSize, hiddenSize, 'f16');
}

function createUniformBuffer(device, recorder, options) {
  const padTokenId = options.padTokenId == null ? 0xFFFFFFFF : options.padTokenId;
  return createUniformBufferWithView(
    'lm_head_argmax_uniforms',
    32,
    (view) => {
      view.setUint32(0, options.vocabSize, true);
      view.setUint32(4, options.hiddenSize, true);
      view.setUint32(8, options.transposeB ? 1 : 0, true);
      view.setUint32(12, options.workgroupsX, true);
      view.setUint32(16, padTokenId, true);
      view.setFloat32(20, options.logitSoftcap, true);
      view.setUint32(24, options.outputIndex, true);
      view.setUint32(28, options.numGroups, true);
    },
    recorder,
    device
  );
}

function planDispatch(device, vocabSize, colsPerWorkgroup) {
  const numGroups = Math.ceil(vocabSize / colsPerWorkgroup);
  const maxWorkgroups = Number.isFinite(device.limits?.maxComputeWorkgroupsPerDimension)
    ? device.limits.maxComputeWorkgroupsPerDimension
    : 65535;
  const workgroupsX = Math.min(numGroups, maxWorkgroups);
  const workgroupsY = Math.ceil(numGroups / workgroupsX);
  if (workgroupsY > maxWorkgroups) {
    throw new Error(
      `[LmHeadArgmax] dispatch exceeds WebGPU workgroup limits: ` +
      `workgroupsX=${workgroupsX}, workgroupsY=${workgroupsY}, max=${maxWorkgroups}.`
    );
  }
  return {
    numGroups,
    workgroups: [workgroupsX, workgroupsY, 1],
    workgroupsX,
  };
}

async function recordLmHeadArgmaxResolved(recorder, inputTensor, weight, options) {
  const phase1Constants = resolveLmHeadArgmaxPipelineConstants(weight.phase1Variant);
  const dispatchPlan = planDispatch(recorder.device, options.vocabSize, phase1Constants.COLS_PER_WG);
  const minOutputBytes = Math.max(4, (options.outputIndex + 1) * 4);
  const outputBuffer = options.outputBuffer ?? acquireBuffer(minOutputBytes, undefined, 'lm_head_argmax_output');
  const ownsOutput = !options.outputBuffer;
  assertOutputBufferSize(outputBuffer, options.outputIndex);

  let tempIndices = null;
  let tempLogits = null;
  let completed = false;
  try {
    tempIndices = acquireBuffer(dispatchPlan.numGroups * 4, undefined, 'lm_head_argmax_temp_indices');
    tempLogits = acquireBuffer(dispatchPlan.numGroups * 4, undefined, 'lm_head_argmax_temp_logits');
    const uniformBuffer = createUniformBuffer(recorder.device, recorder, {
      vocabSize: options.vocabSize,
      hiddenSize: options.hiddenSize,
      transposeB: weight.transposeB,
      workgroupsX: dispatchPlan.workgroupsX,
      padTokenId: options.padTokenId,
      logitSoftcap: options.logitSoftcap,
      outputIndex: options.outputIndex,
      numGroups: dispatchPlan.numGroups,
    });
    const layout = getLmHeadArgmaxBindGroupLayout(recorder.device);
    const entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputTensor.buffer } },
      { binding: 2, resource: { buffer: weight.buffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: tempIndices } },
      { binding: 5, resource: { buffer: tempLogits } },
    ];
    const bindGroup = recorder.device.createBindGroup({
      label: 'lm_head_argmax_bind_group',
      layout,
      entries,
    });
    const phase1Pipeline = await createLmHeadArgmaxPipeline(recorder.device, weight.phase1Variant);
    const phase1Label = weight.dtype === 'q4k'
      ? 'lm_head_argmax_q4k_phase1'
      : 'lm_head_argmax_phase1';
    recordDispatch(
      recorder,
      phase1Pipeline,
      bindGroup,
      dispatchPlan.workgroups,
      phase1Label
    );

    const phase2Pipeline = await createLmHeadArgmaxPipeline(recorder.device, weight.phase2Variant);
    const phase2Label = weight.dtype === 'q4k'
      ? 'lm_head_argmax_q4k_phase2'
      : 'lm_head_argmax_phase2';
    recordDispatch(recorder, phase2Pipeline, bindGroup, [1, 1, 1], phase2Label);

    recorder.trackTemporaryBuffer(tempIndices);
    recorder.trackTemporaryBuffer(tempLogits);
    completed = true;
    return outputBuffer;
  } finally {
    if (!completed) {
      if (tempIndices) releaseBuffer(tempIndices);
      if (tempLogits) releaseBuffer(tempLogits);
      if (ownsOutput) releaseBuffer(outputBuffer);
    }
  }
}

export async function recordLmHeadArgmax(recorder, inputTensor, lmHead, options = {}) {
  if (!recorder?.device) {
    throw new Error('[LmHeadArgmax] CommandRecorder is required.');
  }
  if (inputTensor?.dtype !== 'f32') {
    throw new Error(`[LmHeadArgmax] input tensor must be f32, got "${inputTensor?.dtype ?? 'unknown'}".`);
  }
  validateArgmaxOptions(options);
  const weight = resolveLmHeadWeight(lmHead, options.vocabSize, options.hiddenSize);
  return recordLmHeadArgmaxResolved(recorder, inputTensor, weight, options);
}

export async function recordLmHeadArgmaxF16(recorder, inputTensor, lmHead, options = {}) {
  if (!recorder?.device) {
    throw new Error('[LmHeadArgmax] CommandRecorder is required.');
  }
  if (inputTensor?.dtype !== 'f32') {
    throw new Error(`[LmHeadArgmax] input tensor must be f32, got "${inputTensor?.dtype ?? 'unknown'}".`);
  }
  validateArgmaxOptions(options);
  const weight = resolveLmHeadWeightForDtype(lmHead, options.vocabSize, options.hiddenSize, 'f16');
  return recordLmHeadArgmaxResolved(recorder, inputTensor, weight, options);
}
