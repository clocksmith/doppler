

import { getDevice, getDeviceEpoch, getDeviceLimits, getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { TILE_SIZES } from './constants.js';
import { getKernelThresholds, padToQ4KBlock } from '../../config/schema/index.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { getKernelConfig } from './kernel-configs.js';
import { getPipelineBindGroupLayout } from './pipeline-cache.js';
import { hasRequiredFeatures } from './feature-check.js';
import { dispatchIndirect, recordDispatchIndirect } from './dispatch.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { log, trace } from '../../debug/index.js';
import { getKernelPathAttentionVariant, getKernelPathStrict } from '../../config/kernel-path-loader.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';
import { logKernelSelectionOnce } from '../kernel-selection-log.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { getRequiredVariantMaxKVLen, resolveAttentionPlan } from './attention/plan.js';
import { assertAttentionBindGroupBuffer, dispatchAttentionKernel, executeAttention, executeAttentionBDPA, getPageTableFallbackBuffer, releaseAttentionUniform, resolveAttentionExecution } from './attention/executors.js';

// Track if we've logged the attention tier selection (avoid spam)

function getTieredMaxKVLen() {
  return getRequiredVariantMaxKVLen('attention_tiered', 'decode_tiered_f16', 'attention_tiered.decode_tiered_f16');
}

function getTieredQuantMaxKVLen() {
  return getRequiredVariantMaxKVLen(
    'attention_tiered_quant',
    'decode_tiered_int8_f16kv',
    'attention_tiered_quant.decode_tiered_int8_f16kv'
  );
}

function getContiguousQuantMaxKVLen() {
  return getRequiredVariantMaxKVLen(
    'attention_contiguous_quant',
    'decode_contiguous_turboquant_f16kv',
    'attention_contiguous_quant.decode_contiguous_turboquant_f16kv'
  );
}

class AttentionTieredKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_tiered', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_tiered');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_tiered');
  }
}

class AttentionTieredQuantKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_tiered_quant', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_tiered_quant');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_tiered_quant');
  }
}

class AttentionContiguousQuantKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_contiguous_quant', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_contiguous_quant');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_contiguous_quant');
  }
}

// Track if we've logged chunked kernel selection

export function resolveAttentionPlanForTest(
  seqLen,
  kvLen,
  headDim,
  numHeads,
  kvDtype,
  qDtype,
  sharedLimit,
  caps,
  layerIdx,
  isPaged = false,
  kernelPath = null
) {
  return resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    kvDtype,
    qDtype,
    sharedLimit,
    caps,
    layerIdx,
    isPaged,
    kernelPath
  );
}

function createTieredAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_tiered_uniforms',
    80,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.coldLen, true);
      view.setUint32(16, params.hotLen, true);
      view.setUint32(20, params.seqLen, true);
      view.setFloat32(24, params.scale, true);
      view.setUint32(28, params.causal ? 1 : 0, true);
      view.setUint32(32, params.startPos, true);
      view.setFloat32(36, params.attnSoftcap, true);
      view.setUint32(40, params.slidingWindow, true);
      view.setUint32(44, params.hotWindow, true);
      view.setUint32(48, params.hotStart, true);
      view.setUint32(52, params.coldPageSize, true);
      view.setUint32(56, params.coldLayout ?? 0, true);
      view.setUint32(60, params.hotLayout ?? 1, true);
      view.setUint32(64, 0, true);
    },
    recorder,
    device
  );
}

function createTieredQuantAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_tiered_quant_uniforms',
    64,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.coldLen, true);
      view.setUint32(16, params.hotLen, true);
      view.setUint32(20, params.seqLen, true);
      view.setFloat32(24, params.scale, true);
      view.setUint32(28, params.causal ? 1 : 0, true);
      view.setUint32(32, params.startPos, true);
      view.setFloat32(36, params.attnSoftcap, true);
      view.setUint32(40, params.slidingWindow, true);
      view.setUint32(44, params.hotWindow, true);
      view.setUint32(48, params.hotStart, true);
      view.setUint32(52, params.packedStride, true);
      view.setUint32(56, 0, true);
    },
    recorder,
    device
  );
}

function createContiguousQuantAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  const hasProdFields = params.packedStrideMSE != null;
  const size = hasProdFields ? 64 : 48;
  return createUniformBufferWithView(
    'attention_contiguous_quant_uniforms',
    size,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true);
      view.setUint32(36, params.slidingWindow, true);
      if (hasProdFields) {
        view.setUint32(40, params.packedStrideMSE, true);
        view.setUint32(44, params.packedStrideResidual, true);
        view.setUint32(48, 0, true);
        view.setUint32(52, 0, true);
        view.setUint32(56, 0, true);
        view.setUint32(60, 0, true);
      } else {
        view.setUint32(40, params.packedStride, true);
        view.setUint32(44, 0, true);
      }
    },
    recorder,
    device
  );
}

// -----------------------------------------------------------------------------
// Flash-attention prefill path (head_dim = 256, f16 KV)
// -----------------------------------------------------------------------------
// Two-pass kernel to raise RDNA3 occupancy: pass 1 processes one KV slice per
// workgroup and writes per-split (acc, m, l) partials; pass 2 merges across
// splits with online softmax. Single recorder, two dispatches — queue order
// handles the read-after-write between passes.

// Single-pass flash-attention dispatcher (ORT-style). 7-binding contract —
// same as attention_head256_f16kv. One kernel launch, no reduce pass.
// Workgroups: (num_heads, ceil(seqLen / ORT_FLASH_WG), 1) with ORT_FLASH_WG=64.

async function executeAttentionTiered(
  recorder,
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    coldLen = 0,
    hotLen = 0,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    hotWindow = hotLen,
    hotStart = 0,
    coldPageTable = null,
    coldPageSize = 0,
    coldLayout = 2,
    hotLayout = 1,
  } = options;

  const totalLen = coldLen + hotLen;
  const maxKVLen = getTieredMaxKVLen();
  if (totalLen > maxKVLen) {
    throw new Error(`Tiered attention requires total KV len <= ${maxKVLen} but got ${totalLen}.`);
  }

  const useF16 = Q.dtype === 'f16' && hotK.dtype === 'f16' && coldK.dtype === 'f16';
  const useF16KV = hotK.dtype === 'f16' && coldK.dtype === 'f16';
  const variant = selectKernelRuleValue('attention', 'tieredVariant', { useF16 });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_tiered', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Tiered attention kernel "${variant}" requires unsupported GPU features.`);
  }
  if (!useF16KV) {
    throw new Error('Tiered attention requires f16 KV buffers.');
  }

  const kernel = new AttentionTieredKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_output');

  const uniformBuffer = createTieredAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    coldLen,
    hotLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    hotWindow,
    hotStart,
    coldPageSize,
    coldLayout,
    hotLayout,
  });

  const pageTableBinding = coldPageTable || getPageTableFallbackBuffer(execution.device);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 2, 'hotK', hotK?.buffer, [
    `hotKLabel=${hotK?.label ?? 'unknown'}`,
    `hotKDtype=${hotK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 3, 'hotV', hotV?.buffer, [
    `hotVLabel=${hotV?.label ?? 'unknown'}`,
    `hotVDtype=${hotV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 4, 'coldK', coldK?.buffer, [
    `coldKLabel=${coldK?.label ?? 'unknown'}`,
    `coldKDtype=${coldK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 5, 'coldV', coldV?.buffer, [
    `coldVLabel=${coldV?.label ?? 'unknown'}`,
    `coldVDtype=${coldV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 6, 'output', outputBuf);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 7, 'pageTable', pageTableBinding, [
    `coldLayout=${coldLayout}`,
  ]);
  const bindGroup = execution.device.createBindGroup({
    label: 'attention_tiered_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldK.buffer } },
      { binding: 5, resource: { buffer: coldV.buffer } },
      { binding: 6, resource: { buffer: outputBuf } },
      { binding: 7, resource: { buffer: pageTableBinding } },
    ],
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_output');
}

async function executeAttentionTieredQuant(
  recorder,
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    coldLen = 0,
    hotLen = 0,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    hotWindow = hotLen,
    hotStart = 0,
    packedStride = 0,
    mode = 'int8',
    // TurboQuant additional buffers
    rotationMatrixBuffer = null,
    codebookCentroidsBuffer = null,
    residualKBuffer = null,
    residualVBuffer = null,
    residualNormsKBuffer = null,
    residualNormsVBuffer = null,
    qjlMatrixBuffer = null,
  } = options;

  if (mode === 'turboquant_outlier') {
    throw new Error(
      'TurboQuant outlier attention is not supported yet. ' +
      'Outlier-mode decode kernels are not wired end to end.'
    );
  }

  const isTurboQuant = mode === 'turboquant' || mode === 'turboquant_prod';
  const isProd = mode === 'turboquant_prod';

  const totalLen = coldLen + hotLen;
  const maxKVLen = getTieredQuantMaxKVLen();
  if (totalLen > maxKVLen) {
    throw new Error(`Tiered quant attention requires total KV len <= ${maxKVLen} but got ${totalLen}.`);
  }
  if (!Number.isFinite(packedStride) || packedStride <= 0) {
    throw new Error('Tiered quant attention requires packedStride > 0.');
  }

  if (Q.dtype !== 'f32') {
    throw new Error('Tiered quant attention requires f32 Q.');
  }

  if (isTurboQuant && !rotationMatrixBuffer) {
    throw new Error('TurboQuant tiered quant attention requires rotationMatrixBuffer.');
  }
  if (isTurboQuant && !codebookCentroidsBuffer) {
    throw new Error('TurboQuant tiered quant attention requires codebookCentroidsBuffer.');
  }
  if (isProd && !qjlMatrixBuffer) {
    throw new Error('TurboQuant prod tiered quant attention requires qjlMatrixBuffer.');
  }

  const variant = selectKernelRuleValue('attention', 'tieredQuantVariant', { mode });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_tiered_quant', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Tiered quant attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const kernel = new AttentionTieredQuantKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered_quant variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_quant_output');

  const uniformBuffer = createTieredQuantAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    coldLen,
    hotLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    hotWindow,
    hotStart,
    packedStride,
  });

  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 2, 'hotK', hotK?.buffer, [
    `hotKLabel=${hotK?.label ?? 'unknown'}`,
    `hotKDtype=${hotK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 3, 'hotV', hotV?.buffer, [
    `hotVLabel=${hotV?.label ?? 'unknown'}`,
    `hotVDtype=${hotV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 4, 'coldPackedK', coldPackedK);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 5, 'coldPackedV', coldPackedV);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 6, 'coldScalesK', coldScalesK);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 7, 'coldScalesV', coldScalesV);

  let entries;
  if (isProd) {
    // TurboQuant prod tiered: 16 bindings
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 8, 'residual_k', residualKBuffer);
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 15, 'qjl_matrix', qjlMatrixBuffer);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: residualKBuffer } },
      { binding: 9, resource: { buffer: residualVBuffer } },
      { binding: 10, resource: { buffer: residualNormsKBuffer } },
      { binding: 11, resource: { buffer: residualNormsVBuffer } },
      { binding: 12, resource: { buffer: outputBuf } },
      { binding: 13, resource: { buffer: rotationMatrixBuffer } },
      { binding: 14, resource: { buffer: codebookCentroidsBuffer } },
      { binding: 15, resource: { buffer: qjlMatrixBuffer } },
    ];
  } else if (isTurboQuant) {
    // TurboQuant MSE tiered: 11 bindings
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: outputBuf } },
      { binding: 9, resource: { buffer: rotationMatrixBuffer } },
      { binding: 10, resource: { buffer: codebookCentroidsBuffer } },
    ];
  } else {
    // Standard int4/int8: 9 bindings
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 8, 'output', outputBuf);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: outputBuf } },
    ];
  }

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_tiered_quant_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries,
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_quant_output');
}

export async function runAttentionBDPA(
  Q,
  basisK,
  basisV,
  pagedK,
  pagedV,
  index,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionBDPA(null, Q, basisK, basisV, pagedK, pagedV, index, numHeads, headDim, options);
}

export async function recordAttentionBDPA(
  recorder,
  Q,
  basisK,
  basisV,
  pagedK,
  pagedV,
  index,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionBDPA(recorder, Q, basisK, basisV, pagedK, pagedV, index, numHeads, headDim, options);
}

export async function runAttention(
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttention(null, Q, K, V, mask, numHeads, headDim, options);
}

export async function recordAttention(
  recorder,
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttention(recorder, Q, K, V, mask, numHeads, headDim, options);
}

export async function runAttentionTiered(
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTiered(null, Q, hotK, hotV, coldK, coldV, numHeads, headDim, options);
}

export async function recordAttentionTiered(
  recorder,
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTiered(recorder, Q, hotK, hotV, coldK, coldV, numHeads, headDim, options);
}

export async function runAttentionTieredQuant(
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTieredQuant(
    null,
    Q,
    hotK,
    hotV,
    coldPackedK,
    coldPackedV,
    coldScalesK,
    coldScalesV,
    numHeads,
    headDim,
    options
  );
}

export async function recordAttentionTieredQuant(
  recorder,
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTieredQuant(
    recorder,
    Q,
    hotK,
    hotV,
    coldPackedK,
    coldPackedV,
    coldScalesK,
    coldScalesV,
    numHeads,
    headDim,
    options
  );
}

// =============================================================================
// Contiguous Quantized Attention (TurboQuant for full-attention models)
// =============================================================================

async function executeAttentionContiguousQuant(
  recorder,
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = 0,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    packedStride = 0,
    mode = 'turboquant',
    rotationMatrixBuffer = null,
    codebookCentroidsBuffer = null,
    // Prod-mode additional buffers
    residualKBuffer = null,
    residualVBuffer = null,
    residualNormsKBuffer = null,
    residualNormsVBuffer = null,
    qjlMatrixBuffer = null,
    packedStrideMSE = 0,
    packedStrideResidual = 0,
  } = options;

  const maxKVLen = getContiguousQuantMaxKVLen();
  if (kvLen > maxKVLen) {
    throw new Error(`Contiguous quant attention requires kvLen <= ${maxKVLen} but got ${kvLen}.`);
  }
  if (!Number.isFinite(packedStride) || packedStride <= 0) {
    throw new Error('Contiguous quant attention requires packedStride > 0.');
  }
  if (Q.dtype !== 'f32') {
    throw new Error('Contiguous quant attention requires f32 Q.');
  }
  if (!rotationMatrixBuffer) {
    throw new Error('Contiguous quant attention requires rotationMatrixBuffer.');
  }
  if (!codebookCentroidsBuffer) {
    throw new Error('Contiguous quant attention requires codebookCentroidsBuffer.');
  }

  const isProd = mode === 'turboquant_prod';
  const variant = selectKernelRuleValue('attention', 'contiguousQuantVariant', { mode });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_contiguous_quant', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Contiguous quant attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const kernel = new AttentionContiguousQuantKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_contiguous_quant variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_contiguous_quant_output');

  const uniformParams = isProd
    ? {
      numHeads, numKVHeads, headDim, kvLen, seqLen, scale, causal, startPos,
      attnSoftcap, slidingWindow, packedStrideMSE, packedStrideResidual,
    }
    : {
      numHeads, numKVHeads, headDim, kvLen, seqLen, scale, causal, startPos,
      attnSoftcap, slidingWindow, packedStride,
    };
  const uniformBuffer = createContiguousQuantAttentionUniformBuffer(
    execution.device, execution.recorder, uniformParams
  );

  let entries;
  if (isProd) {
    // Contiguous prod: 14 bindings
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 6, 'residual_k', residualKBuffer);
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 7, 'residual_v', residualVBuffer);
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 13, 'qjl_matrix', qjlMatrixBuffer);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: packedK } },
      { binding: 3, resource: { buffer: packedV } },
      { binding: 4, resource: { buffer: scalesK } },
      { binding: 5, resource: { buffer: scalesV } },
      { binding: 6, resource: { buffer: residualKBuffer } },
      { binding: 7, resource: { buffer: residualVBuffer } },
      { binding: 8, resource: { buffer: residualNormsKBuffer } },
      { binding: 9, resource: { buffer: residualNormsVBuffer } },
      { binding: 10, resource: { buffer: outputBuf } },
      { binding: 11, resource: { buffer: rotationMatrixBuffer } },
      { binding: 12, resource: { buffer: codebookCentroidsBuffer } },
      { binding: 13, resource: { buffer: qjlMatrixBuffer } },
    ];
  } else {
    // Contiguous MSE: 9 bindings
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: packedK } },
      { binding: 3, resource: { buffer: packedV } },
      { binding: 4, resource: { buffer: scalesK } },
      { binding: 5, resource: { buffer: scalesV } },
      { binding: 6, resource: { buffer: outputBuf } },
      { binding: 7, resource: { buffer: rotationMatrixBuffer } },
      { binding: 8, resource: { buffer: codebookCentroidsBuffer } },
    ];
  }

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_contiguous_quant_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries,
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_contiguous_quant_output');
}

export async function runAttentionContiguousQuant(
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionContiguousQuant(
    null, Q, packedK, packedV, scalesK, scalesV, numHeads, headDim, options
  );
}

export async function recordAttentionContiguousQuant(
  recorder,
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionContiguousQuant(
    recorder, Q, packedK, packedV, scalesK, scalesV, numHeads, headDim, options
  );
}
