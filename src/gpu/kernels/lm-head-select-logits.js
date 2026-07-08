import { acquireBuffer, releaseBuffer, uploadData } from '../../memory/buffer-pool.js';
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
import { dispatch } from './dispatch.js';

const WORKGROUP_SIZE = 256;

function getLmHeadSelectBindGroupLayout(device) {
  return getOrCreateBindGroupLayout(
    'lm_head_select_logits_bind_group_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
}

async function createLmHeadSelectPipeline(device) {
  return createPipeline(
    'lm_head_select_logits',
    'f16w_f32a',
    getLmHeadSelectBindGroupLayout(device),
    {
      WORKGROUP_SIZE,
    }
  );
}

function normalizeSelectedTokenIds(tokenIds, vocabSize) {
  if (!Array.isArray(tokenIds) && !ArrayBuffer.isView(tokenIds)) {
    throw new Error('[LmHeadSelect] tokenIds must be an array or typed array.');
  }
  const normalized = Array.from(tokenIds, (value, index) => {
    const tokenId = Number(value);
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
      throw new Error(
        `[LmHeadSelect] tokenIds[${index}] must be an integer in [0, ${vocabSize}), got "${String(value)}".`
      );
    }
    return tokenId;
  });
  if (normalized.length === 0) {
    throw new Error('[LmHeadSelect] tokenIds must not be empty.');
  }
  return normalized;
}

function resolveLmHeadWeight(lmHead, vocabSize, hiddenSize) {
  if (!isWeightBuffer(lmHead)) {
    throw new Error('[LmHeadSelect] selected-token LM head requires a GPU-resident WeightBuffer.');
  }
  const resolved = resolveWeightBufferMaterialization(lmHead, 'f16');
  if (!isWeightBuffer(resolved)) {
    throw new Error('[LmHeadSelect] selected-token LM head requires a single GPU-resident weight buffer.');
  }
  const dtype = getWeightDtype(resolved);
  if (dtype !== 'f16') {
    throw new Error(`[LmHeadSelect] selected-token LM head requires f16 weights, got "${dtype ?? 'unknown'}".`);
  }
  const layout = getLayout(resolved);
  if (layout !== 'row' && layout !== 'column') {
    throw new Error(`[LmHeadSelect] selected-token LM head requires row or column layout, got "${layout ?? 'unknown'}".`);
  }
  const shape = resolved.shape;
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error(`[LmHeadSelect] selected-token LM head requires 2D weights, got [${shape?.join?.(', ') ?? ''}].`);
  }
  const shapeVocab = layout === 'row' ? shape[0] : shape[1];
  const shapeHidden = layout === 'row' ? shape[1] : shape[0];
  if (shapeVocab < vocabSize || shapeHidden !== hiddenSize) {
    throw new Error(
      `[LmHeadSelect] LM head shape mismatch: layout=${layout}, ` +
      `shape=[${shape.join(', ')}], vocab=${vocabSize}, hidden=${hiddenSize}.`
    );
  }
  return {
    buffer: getBuffer(resolved),
    transposeB: layout === 'row',
  };
}

function createUniformBuffer(device, options) {
  return createUniformBufferWithView(
    'lm_head_select_logits_uniforms',
    32,
    (view) => {
      view.setUint32(0, options.hiddenSize, true);
      view.setUint32(4, options.vocabSize, true);
      view.setUint32(8, options.tokenCount, true);
      view.setUint32(12, options.hiddenOffset, true);
      view.setUint32(16, options.transposeB ? 1 : 0, true);
      view.setFloat32(20, options.logitSoftcap, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    null,
    device
  );
}

export async function runLmHeadSelectLogitsF16(inputTensor, lmHead, options = {}) {
  const device = options.device;
  if (!device) {
    throw new Error('[LmHeadSelect] GPU device is required.');
  }
  if (inputTensor?.dtype !== 'f32') {
    throw new Error(`[LmHeadSelect] input tensor must be f32, got "${inputTensor?.dtype ?? 'unknown'}".`);
  }
  const hiddenSize = options.hiddenSize;
  const vocabSize = options.vocabSize;
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`[LmHeadSelect] hiddenSize must be a positive integer, got "${String(hiddenSize)}".`);
  }
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new Error(`[LmHeadSelect] vocabSize must be a positive integer, got "${String(vocabSize)}".`);
  }
  if (options.hiddenOffset === undefined) {
    throw new Error('[LmHeadSelect] hiddenOffset is required.');
  }
  const hiddenOffset = Number(options.hiddenOffset);
  if (!Number.isInteger(hiddenOffset) || hiddenOffset < 0) {
    throw new Error(`[LmHeadSelect] hiddenOffset must be a non-negative integer, got "${String(options.hiddenOffset)}".`);
  }
  if (options.logitSoftcap === undefined) {
    throw new Error('[LmHeadSelect] logitSoftcap is required.');
  }
  const logitSoftcap = Number(options.logitSoftcap);
  if (!Number.isFinite(logitSoftcap) || logitSoftcap < 0) {
    throw new Error(`[LmHeadSelect] logitSoftcap must be a finite non-negative number, got "${String(options.logitSoftcap)}".`);
  }

  const tokenIds = normalizeSelectedTokenIds(options.tokenIds, vocabSize);
  const weight = resolveLmHeadWeight(lmHead, vocabSize, hiddenSize);
  const tokenIdBuffer = acquireBuffer(tokenIds.length * Uint32Array.BYTES_PER_ELEMENT, undefined, 'lm_head_select_token_ids');
  const outputBuffer = acquireBuffer(tokenIds.length * Float32Array.BYTES_PER_ELEMENT, undefined, 'lm_head_select_logits_output');
  let completed = false;
  try {
    uploadData(tokenIdBuffer, Uint32Array.from(tokenIds));
    const uniformBuffer = createUniformBuffer(device, {
      hiddenSize,
      vocabSize,
      tokenCount: tokenIds.length,
      hiddenOffset,
      transposeB: weight.transposeB,
      logitSoftcap,
    });
    const pipeline = await createLmHeadSelectPipeline(device);
    const bindGroup = device.createBindGroup({
      label: 'lm_head_select_logits_bind_group',
      layout: getLmHeadSelectBindGroupLayout(device),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputTensor.buffer } },
        { binding: 2, resource: { buffer: weight.buffer } },
        { binding: 3, resource: { buffer: tokenIdBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
      ],
    });
    dispatch(device, pipeline, bindGroup, [tokenIds.length, 1, 1], 'lm_head_select_logits');
    completed = true;
    return {
      outputBuffer,
      tokenIdBuffer,
      tokenIds,
    };
  } finally {
    if (!completed) {
      releaseBuffer(tokenIdBuffer);
      releaseBuffer(outputBuffer);
    }
  }
}
