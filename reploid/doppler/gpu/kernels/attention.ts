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
import type { OutputBufferOptions } from './types.js';

const DEBUG_KERNELS = typeof window !== 'undefined'
  ? Boolean((window as unknown as { DOPPLER_DEBUG_KERNELS?: boolean }).DOPPLER_DEBUG_KERNELS)
  : false;

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
    console.warn(
      `[Attention] Requested tiled_large but device doesn't support it ` +
      `(headDim=${headDim}, shared=${sharedLimit}). Falling back.`
    );
    tier = null;
  }
  if (tier === 'tiled_small' && !canSmall) {
    console.warn(
      `[Attention] Requested tiled_small but device doesn't support it ` +
      `(headDim=${headDim}, shared=${sharedLimit}). Falling back.`
    );
    tier = null;
  }

  if (!tier) {
    if (canSubgroup) {
      tier = 'subgroup';
      console.log(`[Attention] Using subgroup decode kernel (headDim=${headDim}, hasSubgroups=true)`);
    } else if (canLarge) {
      tier = 'tiled_large';
    } else if (canSmall) {
      tier = 'tiled_small';
    } else if (isDecode) {
      tier = 'streaming';
    } else {
      console.warn(
        `[Attention] No tiled kernel fits prefill (headDim=${headDim}, shared=${sharedLimit}). ` +
        `Falling back to streaming. Expect slow prefill.`
      );
      tier = 'streaming';
    }
  }

  return tier as AttentionTier;
}

function resolveAttentionVariant(
  tier: AttentionTier,
  isDecode: boolean,
  useF16KV: boolean,
  numHeads: number,
  headDim: number
): string {
  const base = isDecode ? 'decode' : 'prefill';
  if (tier === 'subgroup') {
    // decode_subgroup only supports F32 KV cache
    // Fall back to chunked for F16 KV cache (better than streaming for large headDim)
    if (useF16KV) {
      // Use chunked kernel for few heads with large headDim (e.g., Gemma 3: 4 heads × 256 dim)
      // This parallelizes across headDim instead of running single-threaded
      if (numHeads <= 8 && headDim >= 128) {
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
  // For streaming with few heads, prefer chunked kernel
  if (isDecode && useF16KV && numHeads <= 8 && headDim >= 128) {
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
  const variant = resolveAttentionVariant(tier, isDecode, useF16KV, numHeads, headDim);
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
  }
): GPUBuffer {
  return createUniformBufferWithView(
    'attention_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
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
  } = options;

  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  const kvDtype = getBufferDtype(K) || 'f32';
  const plan = resolveAttentionPlan(
    seqLen,
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

  uniformBuffer.destroy();

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
  } = options;

  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  const kvDtype = getBufferDtype(K) || 'f32';
  const plan = resolveAttentionPlan(
    seqLen,
    headDim,
    numHeads,
    attentionKernel,
    kvDtype,
    sharedLimit,
    caps
  );

  if (DEBUG_KERNELS) {
    console.warn(
      `[ATTN] recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, ` +
      `seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`
    );
  }

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
