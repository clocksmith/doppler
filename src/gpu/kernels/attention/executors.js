import { executeQuerySplitK } from './query-splitk.js';
import { executePairedAttention } from '../paired-attention.js';
import { getDeviceLimits, getKernelCapabilities } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { KernelBase } from '../kernel-base.js';
import { withKernelOutput } from '../kernel-execution.js';
import { padToQ4KBlock } from '../../../config/schema/index.js';
import { createUniformBufferWithView } from '../uniform-utils.js';
import { getKernelConfig, getKernelValidator } from '../kernel-configs.js';
import { getPipelineBindGroupLayout } from '../pipeline-cache.js';
import { hasRequiredFeatures } from '../feature-check.js';
import { dispatchIndirect, recordDispatchIndirect } from '../dispatch.js';
import { trace } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import { declareDecodeAttentionLimit, resolveAttentionPlan, resolveOnlineDecodePipelineConstants } from './plan.js';
import {
  F32_BYTES,
  getKvLenFallbackBuffer,
  getPageTableFallbackBuffer,
  AttentionKernel,
  AttentionBDPAKernel,
  createAttentionUniformBuffer,
  createBDPAAttentionUniformBuffer,
  resolveAttentionExecution,
  assertAttentionBindGroupBuffer,
  releaseAttentionUniform,
  dispatchAttentionKernel,
} from './executors-components.js';

export {
  kvLenFallbackBuffer,
  kvLenFallbackBufferEpoch,
  U32_BYTES,
  F32_BYTES,
  getKvLenFallbackBuffer,
  pageTableFallbackBuffer,
  pageTableFallbackBufferEpoch,
  getPageTableFallbackBuffer,
  AttentionKernel,
  AttentionBDPAKernel,
  createAttentionUniformBuffer,
  createBDPAAttentionUniformBuffer,
  resolveAttentionExecution,
  assertAttentionBindGroupBuffer,
  releaseAttentionUniform,
  dispatchAttentionKernel,
} from './executors-components.js';

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
  if ((options.bidirectionalSpanLength ?? 0) > 0) {
    throw new Error('[Attention] BDPA attention has no bidirectional-span mask.');
  }

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
  if (options.useOrtFlashPrefill === true) {
    return executeOrtFlashAttentionPrefill();
  }
  if (Q.precision !== undefined || Q.components !== undefined || Q.elementStrideBytes !== undefined) {
    return executePairedAttention(recorder, Q, K, V, mask, numHeads, headDim, options);
  }
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
  if (options.useFlashPrefill === true && plan.tier !== 'flash_prefill') {
    throw new Error('[FlashAttention] useFlashPrefill requires an execution graph selecting the registered flash prefill kernel.');
  }
  // A span is honored only by kernels whose uniform contract declares it; any other
  // kernel would apply plain causal masking to the span's later keys.
  if (bidirectionalSpanLength > 0 && !getKernelConfig('attention', plan.variant).uniforms?.fields
    ?.some((field) => field.name === 'bidirectional_span_length')) {
    throw new Error(`[Attention] Variant "${plan.variant}" has no bidirectional-span mask; select a span-capable attention kernel.`);
  }
  if (plan.isDecode) declareDecodeAttentionLimit(execution.recorder, plan.variant, kvLen, kvStart, slidingWindow);

  getKernelValidator('attention', plan.variant)?.({
    operation: 'attention', variant: plan.variant, bindings: [Q, K, V, outputBuffer],
    uniforms: { seqLen, numHeads, headDim, kvLen, numKVHeads },
    workgroups: plan.workgroups, constants: null, extraBindings: null,
  });

  if (getKernelConfig('attention', plan.variant).variantMetadata?.querySplitK === true) {
    if (mask != null || indirectBuffer != null) throw new Error('[Query split-K] Explicit masks and indirect dispatch are unsupported.');
    return executeQuerySplitK(recorder, plan.variant, Q, K, V, numHeads, headDim, {
      seqLen, kvLen, numKVHeads, scale, causal, startPos, outputBuffer, attnSoftcap, slidingWindow,
      kvLenBuffer, kvStart, kvLayout, kvPageTable, kvPageSize, bidirectionalSpanLength,
    });
  }

  if (plan.variant === DECODE_GQA_SPLITK_VARIANT) {
    if (bidirectionalSpanLength > 0 || mask != null || indirectBuffer != null || seqLen !== 1
      || Q.dtype !== 'f32' || K.dtype !== 'f32' || V.dtype !== 'f32' || headDim !== 256
      || numHeads !== numKVHeads * 8 || slidingWindow > 0 || attnSoftcap !== 0 || kvStart !== 0
      || (kvLayout != null && kvLayout !== 'contiguous') || !causal || startPos + 1 !== kvLen) {
      throw new Error('[Attention] decode_gqa_splitk requires single-query causal decode, f32 Q/K/V, '
        + 'head_dim 256, eight query heads per KV head and contiguous unwindowed KV.');
    }
    return executeDecodeGqaSplitK(execution, Q, K, V, numHeads, headDim, {
      kvLen, numKVHeads, scale, startPos, outputBuffer, kvLenBuffer,
    });
  }

  // The selected split-K kernel owns a nine-binding, two-pass contract.
  // It cannot use the ordinary seven-binding attention dispatcher.
  if (plan.tier === 'flash_prefill') {
    if (bidirectionalSpanLength > 0 || mask != null) {
      throw new Error('[FlashAttention] bidirectional spans and explicit masks are unsupported.');
    }
    if (indirectBuffer != null || seqLen <= 1 || Q.dtype !== 'f32' || K.dtype !== 'f16' || V.dtype !== 'f16') {
      throw new Error('[FlashAttention] requires direct prefill with f32 Q and f16 K/V.');
    }
    return executeFlashAttentionPrefill(recorder, Q, K, V, numHeads, headDim, {
      seqLen, kvLen, numKVHeads, scale, causal, startPos, outputBuffer,
      attnSoftcap, slidingWindow, kvLenBuffer, kvStart, kvLayout, kvPageTable, kvPageSize,
    });
  }

  if (execution.recorder) {
    trace.attn(0, `recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`);
  }

  const kernel = new AttentionKernel(execution.device);
  const pipelineConstants = resolveOnlineDecodePipelineConstants(
    plan.variant,
    headDim,
    getRuntimeConfig()?.inference?.session?.attentionDecodeOnline,
    { kvLayout, slidingWindow }
  );
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

export const DECODE_GQA_SPLITK_VARIANT = 'decode_gqa_splitk';
const DECODE_GQA_SPLITK_SPLITS = 64;

// Two passes in queue order: per-(split, kv head) partials, then a per-head combine.
// The dispatch grid depends only on the model geometry, never on the position.
async function executeDecodeGqaSplitK(execution, Q, K, V, numHeads, headDim, options) {
  const { kvLen, numKVHeads, scale, startPos, outputBuffer, kvLenBuffer } = options;
  const device = execution.device;
  const kernel = new AttentionKernel(device);
  const constants = { SPLITS: DECODE_GQA_SPLITK_SPLITS };
  const partialPipeline = await kernel.getPipeline(DECODE_GQA_SPLITK_VARIANT, constants);
  const combinePipeline = await kernel.getPipeline(`${DECODE_GQA_SPLITK_VARIANT}_combine`, constants);
  const outputBuf = outputBuffer || acquireBuffer(padToQ4KBlock(numHeads * headDim) * 4, undefined, 'attention_output');
  const partialBytes = numHeads * DECODE_GQA_SPLITK_SPLITS * (headDim + 2) * 4;
  const partials = acquireBuffer(partialBytes, undefined, 'attention_splitk_partials');
  const uniformBuffer = createAttentionUniformBuffer(device, execution.recorder, {
    numHeads, numKVHeads, headDim, kvLen, seqLen: 1, scale, causal: true, startPos,
    attnSoftcap: 0, slidingWindow: 0, kvLenSource: kvLenBuffer ? 1 : 0, kvStart: 0, pageSize: 0, kvLayout: 0,
    bidirectionalSpanStart: 0, bidirectionalSpanLength: 0,
  });
  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device);
  try {
    const partialGroup = device.createBindGroup({
      label: 'attention_splitk_partials_bg',
      layout: getPipelineBindGroupLayout(partialPipeline, 0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: Q.buffer } },
        { binding: 2, resource: { buffer: K.buffer } },
        { binding: 3, resource: { buffer: V.buffer } },
        { binding: 5, resource: { buffer: kvLenBinding } },
        { binding: 7, resource: { buffer: partials } },
      ],
    });
    const combineGroup = device.createBindGroup({
      label: 'attention_splitk_combine_bg',
      layout: getPipelineBindGroupLayout(combinePipeline, 0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 4, resource: { buffer: outputBuf } },
        { binding: 5, resource: { buffer: kvLenBinding } },
        { binding: 7, resource: { buffer: partials } },
      ],
    });
    dispatchAttentionKernel(execution, kernel, partialPipeline, partialGroup, [DECODE_GQA_SPLITK_SPLITS, numKVHeads, 1]);
    dispatchAttentionKernel(execution, kernel, combinePipeline, combineGroup, [numHeads, 1, 1]);
  } finally {
    if (execution.recorder) {
      execution.recorder.trackTemporaryBuffer(partials);
    } else {
      await device.queue.onSubmittedWorkDone();
      releaseBuffer(partials);
    }
    releaseAttentionUniform(execution, uniformBuffer);
  }
  return createTensor(outputBuf, 'f32', [1, numHeads, headDim], 'attention_output');
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
  if (!flashPrefillKernel || flashPrefillKernel.device !== device) {
    flashPrefillKernel = new FlashAttentionPrefillKernel(device);
  }
  return flashPrefillKernel;
}

export function getFlashReduceKernel(device) {
  if (!flashReduceKernel || flashReduceKernel.device !== device) {
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
  return withKernelOutput(recorder, outputBuffer, outputSize, 'attention_flash_output', async (outputBuf) => {
    const intermediates = [];
    const uniforms = [];
    const acquireIntermediate = (bytes, label) => {
      const buffer = acquireBuffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label);
      intermediates.push(buffer);
      return buffer;
    };
    try {

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
        : acquireIntermediate(partialAccBytes, 'flash_partial_acc');
      const partialM = acquireIntermediate(
        partialStatsBytes,
        'flash_partial_m'
      );
      const partialL = acquireIntermediate(
        partialStatsBytes,
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

      uniforms.push(flashUniform);
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


      // Pass 2 — reduce. Skipped on the single-split fast path where pass 1
      // already wrote the final normalised output directly to outputBuf.
      if (!singleSplit) {
        const reduceUniform = createFlashReduceUniformBuffer(device, execution.recorder, {
          numHeads,
          queryLen: seqLen,
          numKvSplits,
        });

        uniforms.push(reduceUniform);
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

      }

      return createTensor(outputBuf, 'f32', [seqLen, numHeads, headDim], 'attention_flash_output');
    } finally {
      // Retain every allocation even when preparing the reduce pass fails.
      if (execution.recorder) {
        for (const buffer of intermediates) execution.recorder.trackTemporaryBuffer(buffer);
      } else {
        try { await device.queue.onSubmittedWorkDone(); }
        finally {
          for (const buffer of intermediates) releaseBuffer(buffer);
          for (const uniform of uniforms) releaseAttentionUniform(execution, uniform);
        }
      }
    }
  });
}

export const ORT_FLASH_WG = 64;

export async function executeOrtFlashAttentionPrefill() {
  throw new Error('[OrtFlashAttention] unsupported: no registered ORT flash kernel. Select a registered attention kernel in the execution graph.');
}
