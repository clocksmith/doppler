/**
 * Attention Kernels
 *
 * Provides optimized attention operations with support for:
 * - Prefill and decode phases
 * - Causal masking
 * - Grouped-query attention (GQA)
 * - Multiple implementation tiers (tiled, streaming)
 * - F16/F32 KV cache support
 */

import { getDevice, getDeviceLimits, getKernelCapabilities } from '../device.js';
import { getBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { KernelBase } from './kernel-base.js';
import { DIMENSION_LIMITS, MEMORY_THRESHOLDS, TILE_SIZES } from './constants.js';
import { createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import type { OutputBufferOptions } from './types.js';
import { log, trace } from '../../debug/index.js';

// Track if we've logged the attention tier selection (avoid spam)
let loggedAttentionTier = false;

/** Attention kernel options */
export interface AttentionOptions extends OutputBufferOptions {
  seqLen?: number;
  kvLen?: number;
  numKVHeads?: number;
  scale?: number;
  causal?: boolean;
  startPos?: number;
  attentionKernel?: string | null;
  slidingWindow?: number;
  /** Gemma 2 attention softcapping: score = tanh(score / softcap) * softcap. 0 = disabled. */
  attnSoftcap?: number;
}

type AttentionTier = 'subgroup' | 'tiled_large' | 'tiled_small' | 'streaming';

interface AttentionPlan {
  tier: AttentionTier;
  variant: string;
  workgroups: number;
  useF16KV: boolean;
  isDecode: boolean;
}

class AttentionKernel extends KernelBase {
  async getPipeline(variant: string): Promise<GPUComputePipeline> {
    return this.getPipelineFor('attention', variant);
  }

  dispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number
  ): void {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention');
  }

  record(
    recorder: CommandRecorder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number
  ): void {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention');
  }
}

function selectAttentionTier(
  headDim: number,
  seqLen: number,
  useF16KV: boolean,
  attentionKernel: string | null,
  sharedLimit: number,
  caps: ReturnType<typeof getKernelCapabilities>
): AttentionTier {
  const isDecode = seqLen === 1;
  const canLarge =
    headDim <= DIMENSION_LIMITS.ATTENTION_LARGE_MAX_HEAD_DIM &&
    sharedLimit >= MEMORY_THRESHOLDS.ATTENTION_LARGE_SHARED;
  const smallRequired = useF16KV
    ? MEMORY_THRESHOLDS.ATTENTION_SMALL_SHARED_F16
    : MEMORY_THRESHOLDS.ATTENTION_SMALL_SHARED_F32;
  const canSmall =
    headDim <= DIMENSION_LIMITS.ATTENTION_SMALL_MAX_HEAD_DIM &&
    sharedLimit >= smallRequired;
  const canSubgroup =
    caps.hasSubgroups &&
    headDim <= DIMENSION_LIMITS.ATTENTION_SUBGROUP_MAX_HEAD_DIM &&
    sharedLimit >= MEMORY_THRESHOLDS.ATTENTION_SUBGROUP_SHARED &&
    isDecode;

  let tier = attentionKernel;

  if (tier === 'tiled_large' && !canLarge) {
    log.warn('Attention', `Requested tiled_large but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}). Falling back.`);
    tier = null;
  }
  if (tier === 'tiled_small' && !canSmall) {
    log.warn('Attention', `Requested tiled_small but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}). Falling back.`);
    tier = null;
  }

  if (!tier) {
    if (canSubgroup) {
      tier = 'subgroup';
      if (!loggedAttentionTier) {
        trace.attn(0, `Using subgroup decode kernel (headDim=${headDim}, hasSubgroups=true)`);
        loggedAttentionTier = true;
      }
    } else if (canLarge) {
      tier = 'tiled_large';
    } else if (canSmall) {
      tier = 'tiled_small';
    } else if (isDecode) {
      tier = 'streaming';
    } else {
      log.warn('Attention', `No tiled kernel fits prefill (headDim=${headDim}, shared=${sharedLimit}). Falling back to streaming. Expect slow prefill.`);
      tier = 'streaming';
    }
  }

  return tier as AttentionTier;
}

// Max KV length supported by chunked attention kernel (limited by shared memory)
const CHUNKED_ATTN_MAX_KV_LEN = 2048;

// Track if we've logged chunked kernel selection
let loggedChunkedKernel = false;

function resolveAttentionVariant(
  tier: AttentionTier,
  isDecode: boolean,
  useF16KV: boolean,
  numHeads: number,
  headDim: number,
  kvLen: number
): string {
  const base = isDecode ? 'decode' : 'prefill';

  // Check if chunked kernel is viable:
  // - Decode only (seqLen=1)
  // - F16 KV cache
  // - Large headDim (parallelizes across dimensions)
  // - KV length within shared memory limit
  const canUseChunked = isDecode && useF16KV && headDim >= 128 && kvLen <= CHUNKED_ATTN_MAX_KV_LEN;

  if (tier === 'subgroup') {
    // decode_subgroup only supports F32 KV cache
    // Fall back to chunked for F16 KV cache (much faster than streaming)
    if (useF16KV) {
      if (canUseChunked) {
        if (!loggedChunkedKernel) {
          trace.attn(0, `Using chunked decode kernel (headDim=${headDim}, numHeads=${numHeads}, f16kv=true)`);
          loggedChunkedKernel = true;
        }
        return 'decode_chunked_f16kv';
      }
      return 'decode_streaming_f16kv';
    }
    return 'decode_subgroup';
  }
  if (tier === 'tiled_large') {
    return base + (useF16KV ? '_f16kv' : '');
  }
  if (tier === 'tiled_small') {
    return `${base}_small${useF16KV ? '_f16kv' : ''}`;
  }
  // For streaming tier, prefer chunked if viable
  if (canUseChunked) {
    if (!loggedChunkedKernel) {
      trace.attn(0, `Using chunked decode kernel (headDim=${headDim}, numHeads=${numHeads}, f16kv=true)`);
      loggedChunkedKernel = true;
    }
    return 'decode_chunked_f16kv';
  }
  return `${base}_streaming${useF16KV ? '_f16kv' : ''}`;
}

function calculateAttentionWorkgroups(tier: AttentionTier, seqLen: number, numHeads: number): number {
  if (tier === 'subgroup') {
    return numHeads;
  }
  if (tier === 'streaming') {
    return seqLen * numHeads;
  }
  if (tier === 'tiled_large') {
    return Math.ceil(seqLen / TILE_SIZES.ATTENTION_LARGE_BLOCK_SIZE) * numHeads;
  }
  return Math.ceil(seqLen / TILE_SIZES.ATTENTION_SMALL_BLOCK_SIZE) * numHeads;
}

function resolveAttentionPlan(
  seqLen: number,
  kvLen: number,
  headDim: number,
  numHeads: number,
  attentionKernel: string | null,
  kvDtype: string,
  sharedLimit: number,
  caps: ReturnType<typeof getKernelCapabilities>
): AttentionPlan {
  const useF16KV = kvDtype === 'f16';
  const tier = selectAttentionTier(headDim, seqLen, useF16KV, attentionKernel, sharedLimit, caps);
  const isDecode = seqLen === 1;
  const variant = resolveAttentionVariant(tier, isDecode, useF16KV, numHeads, headDim, kvLen);
  const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads);

  return { tier, variant, workgroups, useF16KV, isDecode };
}

function createAttentionUniformBuffer(
  device: GPUDevice,
  recorder: CommandRecorder | null,
  params: {
    numHeads: number;
    numKVHeads: number;
    headDim: number;
    kvLen: number;
    seqLen: number;
    scale: number;
    causal: boolean;
    startPos: number;
    attnSoftcap: number;
    slidingWindow: number;
  }
): GPUBuffer {
  return createUniformBufferWithView(
    'attention_uniforms',
    48, // 40 bytes used + 8 padding for 16-byte alignment
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
    },
    recorder,
    device
  );
}

/**
 * Run attention operation
 */
export async function runAttention(
  Q: GPUBuffer,
  K: GPUBuffer,
  V: GPUBuffer,
  mask: GPUBuffer | null,
  numHeads: number,
  headDim: number,
  options: AttentionOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    attentionKernel = null,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
  } = options;

  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  const kvDtype = getBufferDtype(K) || 'f32';
  const plan = resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    attentionKernel,
    kvDtype,
    sharedLimit,
    caps
  );
  const kernel = new AttentionKernel(device);
  const pipeline = await kernel.getPipeline(plan.variant);

  // Create output buffer if not provided
  const outputSize = seqLen * numHeads * headDim * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_output');

  // Create uniform buffer
  const uniformBuffer = createAttentionUniformBuffer(device, null, {
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

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'attention_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q } },
      { binding: 2, resource: { buffer: K } },
      { binding: 3, resource: { buffer: V } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  if (limits && plan.workgroups > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention dispatch requires ${plan.workgroups} workgroups but device limit is ` +
      `${limits.maxComputeWorkgroupsPerDimension}. Reduce prompt length or use streaming attention.`
    );
  }



  kernel.dispatch(pipeline, bindGroup, plan.workgroups);

  releaseUniformBuffer(uniformBuffer);

  return output;
}

/**
 * Record attention operation (batched, no submit)
 */
export async function recordAttention(
  recorder: CommandRecorder,
  Q: GPUBuffer,
  K: GPUBuffer,
  V: GPUBuffer,
  mask: GPUBuffer | null,
  numHeads: number,
  headDim: number,
  options: AttentionOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    attentionKernel = null,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
  } = options;

  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  const kvDtype = getBufferDtype(K) || 'f32';
  const plan = resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    attentionKernel,
    kvDtype,
    sharedLimit,
    caps
  );

  trace.attn(0, `recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`);

  const kernel = new AttentionKernel(device);
  const pipeline = await kernel.getPipeline(plan.variant);

  const outputSize = seqLen * numHeads * headDim * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_output');

  const uniformBuffer = createAttentionUniformBuffer(device, recorder, {
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

  const bindGroup = device.createBindGroup({
    label: 'attention_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q } },
      { binding: 2, resource: { buffer: K } },
      { binding: 3, resource: { buffer: V } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  if (limits && plan.workgroups > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention dispatch requires ${plan.workgroups} workgroups but device limit is ` +
      `${limits.maxComputeWorkgroupsPerDimension}. Reduce prompt length or use streaming attention.`
    );
  }

  kernel.record(recorder, pipeline, bindGroup, plan.workgroups);

  return output;
}
