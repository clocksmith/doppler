import { getDevice, getDeviceEpoch, getDeviceLimits, getKernelCapabilities } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { KernelBase } from '../kernel-base.js';
import { TILE_SIZES } from '../constants.js';
import { getKernelThresholds, padToQ4KBlock } from '../../../config/schema/index.js';
import { createUniformBufferWithView } from '../uniform-utils.js';
import { getKernelConfig } from '../kernel-configs.js';
import { getPipelineBindGroupLayout } from '../pipeline-cache.js';
import { hasRequiredFeatures } from '../feature-check.js';
import { dispatchIndirect, recordDispatchIndirect } from '../dispatch.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';
import { log, trace } from '../../../debug/index.js';
import { getKernelPathAttentionVariant, getKernelPathStrict } from '../../../config/kernel-path-loader.js';
import { selectRuleValue as selectKernelRuleValue } from '../rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../../rules/rule-registry.js';
import { logKernelSelectionOnce } from '../../kernel-selection-log.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import { getRequiredVariantMaxKVLen, resolveAttentionPlan } from './plan.js';

export let kvLenFallbackBuffer = null;

export let kvLenFallbackBufferEpoch = -1;

export const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;

export const F32_BYTES = Float32Array.BYTES_PER_ELEMENT;

export function getKvLenFallbackBuffer(device) {
  const epoch = getDeviceEpoch();
  if (!kvLenFallbackBuffer || kvLenFallbackBufferEpoch !== epoch) {
    kvLenFallbackBuffer = device.createBuffer({
      label: 'attention_kv_len_fallback',
      size: U32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(kvLenFallbackBuffer, 0, new Uint32Array([0]));
    kvLenFallbackBufferEpoch = epoch;
  }
  return kvLenFallbackBuffer;
}

export let pageTableFallbackBuffer = null;

export let pageTableFallbackBufferEpoch = -1;

export function getPageTableFallbackBuffer(device) {
  const epoch = getDeviceEpoch();
  if (!pageTableFallbackBuffer || pageTableFallbackBufferEpoch !== epoch) {
    pageTableFallbackBuffer = device.createBuffer({
      label: 'attention_page_table_fallback',
      size: U32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pageTableFallbackBuffer, 0, new Uint32Array([0]));
    pageTableFallbackBufferEpoch = epoch;
  }
  return pageTableFallbackBuffer;
}

export class AttentionKernel extends KernelBase {

  async getPipeline(variant, constants = null) {
    return this.getPipelineFor('attention', variant, null, constants);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention');
  }
}

export function resolveOnlineDecodePipelineConstants(variant, headDim, options = {}) {
  if (!variant?.startsWith('decode_online')) {
    return null;
  }
  const policy = getRuntimeConfig()?.inference?.session?.attentionDecodeOnline;
  const workgroupSize = policy?.workgroupSize;
  const constants = {};
  if (workgroupSize != null) {
    if (workgroupSize !== 128 && workgroupSize !== 256) {
      throw new Error(`attentionDecodeOnline.workgroupSize must be 128 or 256, got ${workgroupSize}.`);
    }
    if (headDim > workgroupSize * 2) {
      throw new Error(
        `attentionDecodeOnline.workgroupSize=${workgroupSize} cannot cover headDim=${headDim}; ` +
        `requires headDim <= ${workgroupSize * 2}.`
      );
    }
    constants.WORKGROUP_SIZE = workgroupSize;
  }
  if (
    variant === 'decode_online_head256_f16kv_output_gate'
    && policy?.useDirectContiguousKVLayout === true
    && options.kvLayout === 'contiguous'
    && options.slidingWindow === 0
  ) {
    constants.USE_DIRECT_KV_LAYOUT = true;
  }
  return Object.keys(constants).length > 0 ? constants : null;
}

export class AttentionBDPAKernel extends KernelBase {
  async getPipeline(variant) {
    return this.getPipelineFor('attention_bdpa', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_bdpa');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_bdpa');
  }
}

export function createAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_uniforms',
    80,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true); // Gemma 2: 50.0, 0 = disabled
      view.setUint32(36, params.slidingWindow, true); // Sliding window size, 0 = disabled
      view.setUint32(40, params.kvLenSource, true); // 0 = uniform kvLen, 1 = buffer
      view.setUint32(44, params.kvStart ?? 0, true);
      view.setUint32(48, params.pageSize ?? 0, true);
      view.setUint32(52, params.kvLayout ?? 0, true);
      view.setUint32(56, params.bidirectionalSpanStart ?? 0, true);
      view.setUint32(60, params.bidirectionalSpanLength ?? 0, true);
      view.setUint32(64, 0, true);
      view.setUint32(68, 0, true);
      view.setUint32(72, 0, true);
    },
    recorder,
    device
  );
}

export function createBDPAAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_bdpa_uniforms',
    64,
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
      view.setUint32(40, 0, true); // padding
      view.setUint32(44, 0, true); // padding
      view.setUint32(48, 0, true); // padding
      view.setUint32(52, 0, true); // padding
      view.setUint32(56, 0, true); // padding
      view.setUint32(60, 0, true); // padding
    },
    recorder,
    device
  );
}

export function resolveAttentionExecution(recorder) {
  return {
    recorder: recorder || null,
    device: recorder?.device || getDevice(),
  };
}

export function assertAttentionBindGroupBuffer(kernelName, variant, bindingIndex, bindingLabel, buffer, details = []) {
  const isGpuBuffer = buffer && (
    typeof GPUBuffer === 'undefined'
      ? true
      : buffer instanceof GPUBuffer
  );
  if (isGpuBuffer) {
    return;
  }
  const detailText = details.filter(Boolean).join(', ');
  throw new Error(
    `[${kernelName}] variant="${variant}" binding ${bindingIndex} "${bindingLabel}" requires a GPUBuffer` +
    (detailText ? ` (${detailText})` : '') +
    '.'
  );
}

export function releaseAttentionUniform(execution, uniformBuffer) {
  if (!execution.recorder) {
    releaseUniformBuffer(uniformBuffer);
  }
}

export function dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, workgroups) {
  if (execution.recorder) {
    kernel.record(execution.recorder, pipeline, bindGroup, workgroups);
    return;
  }
  kernel.dispatch(pipeline, bindGroup, workgroups);
}

export async function executeAttentionBDPA(
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
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    ropeCos = null,
    ropeSin = null,
  } = options;

  if (seqLen !== 1) {
    throw new Error(`BDPA attention currently supports decode only (seqLen=1), got seqLen=${seqLen}.`);
  }
  if (Q.dtype !== 'f16' || basisK.dtype !== 'f16' || basisV.dtype !== 'f16') {
    throw new Error(`BDPA attention requires f16 Q/basis tensors; got Q=${Q.dtype}, basisK=${basisK.dtype}, basisV=${basisV.dtype}.`);
  }
  if (!(ropeCos instanceof GPUBuffer) || !(ropeSin instanceof GPUBuffer)) {
    throw new Error('BDPA attention requires GPU ropeCos/ropeSin buffers.');
  }

  const variant = 'decode_bdpa_f16';
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_bdpa', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`BDPA attention kernel "${variant}" requires unsupported GPU features.`);
  }
  const maxKVLen = config.variantMetadata?.maxKVLen;
  if (Number.isFinite(maxKVLen) && kvLen > maxKVLen) {
    throw new Error(`BDPA attention requires kvLen <= ${maxKVLen} but got ${kvLen}.`);
  }

  const kernel = new AttentionBDPAKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_bdpa variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_bdpa_output');

  const uniformBuffer = createBDPAAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
  });

  assertAttentionBindGroupBuffer('attention_bdpa', variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 2, 'basisK', basisK?.buffer, [
    `basisKLabel=${basisK?.label ?? 'unknown'}`,
    `basisKDtype=${basisK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 3, 'basisV', basisV?.buffer, [
    `basisVLabel=${basisV?.label ?? 'unknown'}`,
    `basisVDtype=${basisV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 4, 'pagedK', pagedK);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 5, 'pagedV', pagedV);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 6, 'index', index);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 7, 'ropeCos', ropeCos);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 8, 'ropeSin', ropeSin);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 9, 'output', outputBuf);

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_bdpa_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: basisK.buffer } },
      { binding: 3, resource: { buffer: basisV.buffer } },
      { binding: 4, resource: { buffer: pagedK } },
      { binding: 5, resource: { buffer: pagedV } },
      { binding: 6, resource: { buffer: index } },
      { binding: 7, resource: { buffer: ropeCos } },
      { binding: 8, resource: { buffer: ropeSin } },
      { binding: 9, resource: { buffer: outputBuf } },
    ],
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_bdpa_output');
}

export async function executeAttention(
  recorder,
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    bidirectionalSpanStart = 0,
    bidirectionalSpanLength = 0,
    startPos = 0,
    layerIdx,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    kvLenBuffer = null,
    indirectBuffer = null,
    indirectOffset = 0,
    kvStart = 0,
    kvLayout = 'contiguous',
    kvPageTable = null,
    kvPageSize = 0,
    kernelPath = null,
    outputGate = null,
  } = options;
  if (!Number.isFinite(bidirectionalSpanStart) || Math.floor(bidirectionalSpanStart) !== bidirectionalSpanStart || bidirectionalSpanStart < 0) {
    throw new Error(`Attention bidirectionalSpanStart must be a non-negative integer, got ${bidirectionalSpanStart}.`);
  }
  if (!Number.isFinite(bidirectionalSpanLength) || Math.floor(bidirectionalSpanLength) !== bidirectionalSpanLength || bidirectionalSpanLength < 0) {
    throw new Error(`Attention bidirectionalSpanLength must be a non-negative integer, got ${bidirectionalSpanLength}.`);
  }
  if (bidirectionalSpanLength > 0 && (bidirectionalSpanStart + bidirectionalSpanLength) > (kvStart + kvLen)) {
    throw new Error(
      `Attention bidirectional span [${bidirectionalSpanStart}, ${bidirectionalSpanStart + bidirectionalSpanLength}) ` +
      `exceeds KV extent [${kvStart}, ${kvStart + kvLen}).`
    );
  }

  // ORT-style single-pass flash attention (adapted from microsoft/onnxruntime
  // flash_attention.wgsl.template). Single kernel, no reduce pass — each WG
  // handles one (head, query-tile) and processes all K online. Gated by
  // useOrtFlashPrefill; takes precedence over the split+reduce flash path.
  if (
    options.useOrtFlashPrefill === true
    && headDim === FLASH_HEAD_DIM
    && seqLen > 1
    && kvLayout === 'contiguous'
    && bidirectionalSpanLength === 0
    && indirectBuffer == null
    && K?.dtype === 'f16'
    && V?.dtype === 'f16'
  ) {
    return executeOrtFlashAttentionPrefill(recorder, Q, K, V, numHeads, headDim, {
      seqLen,
      kvLen,
      numKVHeads,
      scale,
      causal,
      startPos,
      outputBuffer,
      attnSoftcap,
      slidingWindow,
      kvLenBuffer,
      kvStart,
      kvLayout,
      kvPageTable,
      kvPageSize,
    });
  }

  // Flash-attention prefill path: raises RDNA3 occupancy via KV-axis workgroup
  // splitting + online-softmax reduction. Gated by options.useFlashPrefill so
  // callers opt in deliberately (runtime config flag). Conservative conditions:
  // head_dim=256, prefill (seqLen>1), contiguous KV, no bidirectional span.
  if (!globalThis.__DOPPLER_FLASH_TRACE2__ && options.useFlashPrefill === true) {
    globalThis.__DOPPLER_FLASH_TRACE2__ = true;
    if (typeof process !== 'undefined' && process?.stderr?.write) process.stderr.write('[FLASH2] useFlash=' + options.useFlashPrefill
      + ' hd=' + headDim + '/' + FLASH_HEAD_DIM
      + ' seq=' + seqLen + ' kvL=' + kvLayout
      + ' biDir=' + bidirectionalSpanLength
      + ' indirect=' + (indirectBuffer == null ? 'n' : 'y')
      + ' K=' + K?.dtype + ' V=' + V?.dtype + '\n');
  }
  if (
    options.useFlashPrefill === true
    && headDim === FLASH_HEAD_DIM
    && seqLen > 1
    && kvLayout === 'contiguous'
    && bidirectionalSpanLength === 0
    && indirectBuffer == null
    && K?.dtype === 'f16'
    && V?.dtype === 'f16'
  ) {
    return executeFlashAttentionPrefill(recorder, Q, K, V, numHeads, headDim, {
      seqLen,
      kvLen,
      numKVHeads,
      scale,
      causal,
      startPos,
      outputBuffer,
      attnSoftcap,
      slidingWindow,
      kvLenBuffer,
      kvStart,
      kvLayout,
      kvPageTable,
      kvPageSize,
    });
  }

  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();

  const kvDtype = K.dtype;
  const qDtype = Q.dtype;
  const isPaged = kvLayout === 'paged';
  const plan = resolveAttentionPlan(
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

  if (execution.recorder) {
    trace.attn(0, `recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`);
  }

  const kernel = new AttentionKernel(execution.device);
  const pipelineConstants = resolveOnlineDecodePipelineConstants(plan.variant, headDim, {
    kvLayout,
    slidingWindow,
  });
  const usesOutputGateFusion = plan.variant === 'decode_online_head256_f16kv_output_gate';
  const pipeline = await kernel.getPipeline(plan.variant, pipelineConstants);

  const outputConfig = getKernelConfig('attention', plan.variant);
  const outputDtype = outputConfig.outputDtype;
  if (!outputDtype) {
    if (execution.recorder) {
      throw new Error(`Kernel config missing outputDtype for attention variant "${plan.variant}".`);
    }
    throw new Error(`[Attention] outputDtype is required for variant "${plan.variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_output');

  const uniformBuffer = createAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    kvLenSource: kvLenBuffer ? 1 : 0,
    kvStart,
    pageSize: kvPageSize,
    kvLayout: kvLayout === 'paged' ? 2 : (kvLayout === 'ring' ? 1 : 0),
    bidirectionalSpanStart,
    bidirectionalSpanLength,
  });

  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(execution.device);
  const pageTableBinding = kvPageTable || getPageTableFallbackBuffer(execution.device);
  const outputGateBinding = usesOutputGateFusion ? outputGate?.buffer : null;
  if (usesOutputGateFusion) {
    if (outputGate?.dtype !== 'f32') {
      throw new Error(`[Attention] outputGate fusion requires f32 gate tensor; got ${String(outputGate?.dtype)}.`);
    }
    const requiredGateBytes = numHeads * headDim * F32_BYTES;
    const actualGateBytes = outputGate?.buffer?.size;
    if (Number.isFinite(actualGateBytes) && actualGateBytes < requiredGateBytes) {
      throw new Error(
        `[Attention] outputGate fusion requires at least ${requiredGateBytes} gate bytes; got ${actualGateBytes}.`
      );
    }
  }
  assertAttentionBindGroupBuffer('attention', plan.variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention', plan.variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention', plan.variant, 2, 'K', K?.buffer, [
    `KLabel=${K?.label ?? 'unknown'}`,
    `KDtype=${K?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention', plan.variant, 3, 'V', V?.buffer, [
    `VLabel=${V?.label ?? 'unknown'}`,
    `VDtype=${V?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention', plan.variant, 4, 'output', outputBuf);
  assertAttentionBindGroupBuffer('attention', plan.variant, 5, 'kvLen', kvLenBinding);
  assertAttentionBindGroupBuffer('attention', plan.variant, 6, 'pageTable', pageTableBinding, [
    `kvLayout=${kvLayout}`,
  ]);
  if (usesOutputGateFusion) {
    assertAttentionBindGroupBuffer('attention', plan.variant, 7, 'outputGate', outputGateBinding, [
      `useOutputGateFusion=${usesOutputGateFusion}`,
    ]);
  }
  const bindGroupEntries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: Q.buffer } },
    { binding: 2, resource: { buffer: K.buffer } },
    { binding: 3, resource: { buffer: V.buffer } },
    { binding: 4, resource: { buffer: outputBuf } },
    { binding: 5, resource: { buffer: kvLenBinding } },
    { binding: 6, resource: { buffer: pageTableBinding } },
  ];
  if (usesOutputGateFusion) {
    bindGroupEntries.push({ binding: 7, resource: { buffer: outputGateBinding } });
  }
  const bindGroup = execution.device.createBindGroup({
    label: 'attention_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries: bindGroupEntries,
  });

  if (!indirectBuffer && limits && plan.workgroups > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention dispatch requires ${plan.workgroups} workgroups but device limit is ` +
      `${limits.maxComputeWorkgroupsPerDimension}. Reduce prompt length or use streaming attention.`
    );
  }

  if (indirectBuffer) {
    if (execution.recorder) {
      recordDispatchIndirect(execution.recorder, pipeline, bindGroup, indirectBuffer, indirectOffset, 'attention');
    } else {
      dispatchIndirect(execution.device, pipeline, bindGroup, indirectBuffer, indirectOffset, 'attention');
    }
  } else {
    dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, plan.workgroups);
  }

  releaseAttentionUniform(execution, uniformBuffer);

  const outputTensor = createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_output');
  outputTensor.outputGateFused = usesOutputGateFusion;
  return outputTensor;
}

export const FLASH_BLOCK_SIZE = 32;

export const FLASH_HEAD_DIM = 256;

export const FLASH_HEAD_DIM_VECS = 64;

export const FLASH_REDUCE_WG = 64;

export let flashPrefillKernel = null;

export let flashReduceKernel = null;

export class FlashAttentionPrefillKernel extends KernelBase {
  async getPipeline(variant) {
    return this.getPipelineFor('attention', variant);
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention');
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention');
  }
}

export function getFlashPrefillKernel(device) {
  if (!flashPrefillKernel) {
    flashPrefillKernel = new FlashAttentionPrefillKernel(device);
  }
  return flashPrefillKernel;
}

export function getFlashReduceKernel(device) {
  if (!flashReduceKernel) {
    flashReduceKernel = new FlashAttentionPrefillKernel(device);
  }
  return flashReduceKernel;
}

export function createFlashAttentionUniformBuffer(device, recorder, params) {
  // Layout mirrors the flash kernel's Uniforms struct (see
  // attention_prefill_flash_head256_f16kv.wgsl). 64 bytes total.
  return createUniformBufferWithView(
    'attention_flash_uniforms',
    64,
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
      view.setUint32(40, params.kvLenSource, true);
      view.setUint32(44, params.kvStart ?? 0, true);
      view.setUint32(48, params.pageSize ?? 0, true);
      view.setUint32(52, params.kvLayout ?? 0, true);
      view.setUint32(56, params.numKvSplits, true);
      view.setUint32(60, 0, true);
    },
    recorder,
    device
  );
}

export function createFlashReduceUniformBuffer(device, recorder, params) {
  return createUniformBufferWithView(
    'attention_flash_reduce_uniforms',
    16,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.queryLen, true);
      view.setUint32(8, params.numKvSplits, true);
      view.setUint32(12, 0, true);
    },
    recorder,
    device
  );
}

export function chooseFlashNumKvSplits(kvLen) {
  // Target roughly 32 workgroups × num_heads × num_kv_splits ≈ 4x RDNA3
  // compute-unit count. Keep at least 2× FLASH_BLOCK_SIZE KV positions per
  // split so each workgroup has enough work to amortise dispatch overhead.
  // Short prefills (kvLen ≤ 2 × BLOCK_SIZE) take the single-split fast path
  // which skips the reduce pass entirely.
  if (kvLen <= 2 * FLASH_BLOCK_SIZE) return 1;
  const maxSplits = Math.min(8, Math.floor(kvLen / (2 * FLASH_BLOCK_SIZE)));
  return Math.max(1, maxSplits);
}

export async function executeFlashAttentionPrefill(recorder, Q, K, V, numHeads, headDim, options = {}) {
  if (headDim !== FLASH_HEAD_DIM) {
    throw new Error(`[FlashAttention] headDim must be ${FLASH_HEAD_DIM}, got ${headDim}.`);
  }
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    kvLenBuffer = null,
    kvStart = 0,
    kvLayout = 'contiguous',
    kvPageTable = null,
    kvPageSize = 0,
  } = options;

  if (kvLayout !== 'contiguous') {
    throw new Error(`[FlashAttention] kvLayout must be "contiguous", got "${kvLayout}".`);
  }

  const device = execution.device;
  const numQueryBlocks = Math.max(1, Math.ceil(seqLen / FLASH_BLOCK_SIZE));
  const numKvSplits = chooseFlashNumKvSplits(kvLen);
  const singleSplit = numKvSplits === 1;

  // Final output buffer (always allocated; bound to slot 4 on the single-split
  // fast path where the kernel writes normalised output directly).
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_flash_output');

  // Intermediate buffers for the multi-split path. On the single-split fast
  // path we bypass the reduce pass entirely and bind the output buffer in
  // slot 4 (partial_acc slot) while m/l bindings get tiny stub buffers — the
  // kernel skips writes to them.
  const partialAccBytes = singleSplit
    ? 4
    : numQueryBlocks * numHeads * numKvSplits * FLASH_BLOCK_SIZE * FLASH_HEAD_DIM * 4;
  const partialStatsBytes = singleSplit
    ? 4
    : numQueryBlocks * numHeads * numKvSplits * FLASH_BLOCK_SIZE * 4;
  const partialAcc = singleSplit
    ? outputBuf
    : acquireBuffer(partialAccBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'flash_partial_acc');
  const partialM = acquireBuffer(
    partialStatsBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    'flash_partial_m'
  );
  const partialL = acquireBuffer(
    partialStatsBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    'flash_partial_l'
  );

  // Pass 1 uniforms + dispatch.
  const flashUniform = createFlashAttentionUniformBuffer(device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    kvLenSource: kvLenBuffer ? 1 : 0,
    kvStart,
    pageSize: kvPageSize,
    kvLayout: 0, // contiguous only for now
    numKvSplits,
  });

  const flashKernel = getFlashPrefillKernel(device);
  const flashPipeline = await flashKernel.getPipeline('prefill_flash_head256_f16kv');
  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device);
  const pageTableBinding = kvPageTable || getPageTableFallbackBuffer(device);

  const flashBindGroup = device.createBindGroup({
    label: 'attention_flash_prefill_bg',
    layout: flashPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: flashUniform } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: K.buffer } },
      { binding: 3, resource: { buffer: V.buffer } },
      { binding: 4, resource: { buffer: partialAcc } },
      { binding: 5, resource: { buffer: partialM } },
      { binding: 6, resource: { buffer: partialL } },
      { binding: 7, resource: { buffer: kvLenBinding } },
      { binding: 8, resource: { buffer: pageTableBinding } },
    ],
  });

  dispatchAttentionKernel(
    execution,
    flashKernel,
    flashPipeline,
    flashBindGroup,
    numQueryBlocks * numHeads * numKvSplits
  );
  releaseAttentionUniform(execution, flashUniform);

  // Pass 2 — reduce. Skipped on the single-split fast path where pass 1
  // already wrote the final normalised output directly to outputBuf.
  if (!singleSplit) {
    const reduceUniform = createFlashReduceUniformBuffer(device, execution.recorder, {
      numHeads,
      queryLen: seqLen,
      numKvSplits,
    });

    const reduceKernel = getFlashReduceKernel(device);
    const reducePipeline = await reduceKernel.getPipeline('prefill_flash_reduce');

    const reduceBindGroup = device.createBindGroup({
      label: 'attention_flash_reduce_bg',
      layout: reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: reduceUniform } },
        { binding: 1, resource: { buffer: partialAcc } },
        { binding: 2, resource: { buffer: partialM } },
        { binding: 3, resource: { buffer: partialL } },
        { binding: 4, resource: { buffer: outputBuf } },
      ],
    });

    const totalQh = seqLen * numHeads;
    const reduceWgX = Math.ceil(totalQh / FLASH_REDUCE_WG);
    dispatchAttentionKernel(
      execution,
      reduceKernel,
      reducePipeline,
      reduceBindGroup,
      [reduceWgX, FLASH_HEAD_DIM_VECS, 1]
    );
    releaseAttentionUniform(execution, reduceUniform);
  }

  // Release intermediate buffers via the recorder's deferred cleanup so GPU
  // work completes before they re-enter the pool. On the single-split path
  // partialAcc IS the output buffer, so we skip it.
  const intermediates = singleSplit ? [partialM, partialL] : [partialAcc, partialM, partialL];
  if (execution.recorder) {
    for (const buf of intermediates) {
      execution.recorder.trackTemporaryBuffer(buf);
    }
  } else {
    device.queue.onSubmittedWorkDone().then(() => {
      for (const buf of intermediates) {
        releaseBuffer(buf);
      }
    });
  }

  return createTensor(outputBuf, 'f32', [seqLen, numHeads, headDim], 'attention_flash_output');
}

export const ORT_FLASH_WG = 64;

export async function executeOrtFlashAttentionPrefill(recorder, Q, K, V, numHeads, headDim, options = {}) {
  if (headDim !== FLASH_HEAD_DIM) {
    throw new Error(`[OrtFlashAttention] headDim must be ${FLASH_HEAD_DIM}, got ${headDim}.`);
  }
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    kvLenBuffer = null,
    kvStart = 0,
    kvLayout = 'contiguous',
    kvPageTable = null,
    kvPageSize = 0,
  } = options;

  if (kvLayout !== 'contiguous') {
    throw new Error(`[OrtFlashAttention] kvLayout must be "contiguous", got "${kvLayout}".`);
  }

  const device = execution.device;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_ort_flash_output');

  const uniform = createAttentionUniformBuffer(device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    kvLenSource: kvLenBuffer ? 1 : 0,
    kvStart,
    pageSize: kvPageSize,
    kvLayout: 0, // contiguous only
  });

  const kernel = new AttentionKernel(device);
  const pipeline = await kernel.getPipeline('prefill_flash_ort_head256_f16kv');
  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device);
  const pageTableBinding = kvPageTable || getPageTableFallbackBuffer(device);

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_ort_flash_prefill_bg',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: K.buffer } },
      { binding: 3, resource: { buffer: V.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: kvLenBinding } },
      { binding: 6, resource: { buffer: pageTableBinding } },
    ],
  });

  const numSeqTiles = Math.max(1, Math.ceil(seqLen / ORT_FLASH_WG));
  const workgroups = [numHeads, numSeqTiles, 1];

  if (execution.recorder) {
    kernel.record(execution.recorder, pipeline, bindGroup, workgroups);
  } else {
    kernel.dispatch(pipeline, bindGroup, workgroups);
  }

  if (uniform) releaseUniformBuffer(uniform);
  return createTensor(outputBuf, 'f32', [seqLen, numHeads, headDim], 'attention_ort_flash_output');
}
