import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
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
    {
      WORKGROUP_SIZE,
      COLS_PER_WG,
      THREADS_PER_COL,
    }
  );
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

function resolveLmHeadWeight(lmHead, vocabSize, hiddenSize) {
  if (!isWeightBuffer(lmHead)) {
    throw new Error('[LmHeadArgmax] LM head fusion requires a GPU-resident WeightBuffer.');
  }
  const resolved = resolveWeightBufferMaterialization(lmHead, 'f16');
  if (!isWeightBuffer(resolved)) {
    throw new Error('[LmHeadArgmax] LM head fusion requires a single GPU-resident weight buffer.');
  }
  const dtype = getWeightDtype(resolved);
  if (dtype !== 'f16') {
    throw new Error(`[LmHeadArgmax] LM head fusion requires f16 weights, got "${dtype ?? 'unknown'}".`);
  }
  const layout = getLayout(resolved);
  if (layout !== 'row' && layout !== 'column') {
    throw new Error(`[LmHeadArgmax] LM head fusion requires row or column layout, got "${layout ?? 'unknown'}".`);
  }
  const shape = resolved.shape;
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error(`[LmHeadArgmax] LM head fusion requires 2D weights, got [${shape?.join?.(', ') ?? ''}].`);
  }
  const shapeVocab = layout === 'row' ? shape[0] : shape[1];
  const shapeHidden = layout === 'row' ? shape[1] : shape[0];
  if (shapeVocab < vocabSize || shapeHidden !== hiddenSize) {
    throw new Error(
      `[LmHeadArgmax] LM head shape mismatch: layout=${layout}, ` +
      `shape=[${shape.join(', ')}], vocab=${vocabSize}, hidden=${hiddenSize}.`
    );
  }
  return {
    buffer: getBuffer(resolved),
    transposeB: layout === 'row',
  };
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

function planDispatch(device, vocabSize) {
  const numGroups = Math.ceil(vocabSize / COLS_PER_WG);
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

export async function recordLmHeadArgmaxF16(recorder, inputTensor, lmHead, options = {}) {
  if (!recorder?.device) {
    throw new Error('[LmHeadArgmax] CommandRecorder is required.');
  }
  if (inputTensor?.dtype !== 'f32') {
    throw new Error(`[LmHeadArgmax] input tensor must be f32, got "${inputTensor?.dtype ?? 'unknown'}".`);
  }
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

  const device = recorder.device;
  const weight = resolveLmHeadWeight(lmHead, vocabSize, hiddenSize);
  const dispatchPlan = planDispatch(device, vocabSize);
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
    const uniformBuffer = createUniformBuffer(device, recorder, {
      vocabSize,
      hiddenSize,
      transposeB: weight.transposeB,
      workgroupsX: dispatchPlan.workgroupsX,
      padTokenId: options.padTokenId,
      logitSoftcap: options.logitSoftcap,
      outputIndex: options.outputIndex,
      numGroups: dispatchPlan.numGroups,
    });
    const layout = getLmHeadArgmaxBindGroupLayout(device);
    const entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputTensor.buffer } },
      { binding: 2, resource: { buffer: weight.buffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: tempIndices } },
      { binding: 5, resource: { buffer: tempLogits } },
    ];
    const bindGroup = device.createBindGroup({
      label: 'lm_head_argmax_bind_group',
      layout,
      entries,
    });
    const phase1Pipeline = await createLmHeadArgmaxPipeline(device, 'phase1_f16w_f32a');
    recordDispatch(
      recorder,
      phase1Pipeline,
      bindGroup,
      dispatchPlan.workgroups,
      'lm_head_argmax_phase1'
    );

    const phase2Pipeline = await createLmHeadArgmaxPipeline(device, 'phase2');
    recordDispatch(recorder, phase2Pipeline, bindGroup, [1, 1, 1], 'lm_head_argmax_phase2');

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
