/**
 * Kernel Utilities - Shared utilities for kernel management
 *
 * Provides shader loading, compilation, caching, and device capability checking.
 */

import { getDevice, getKernelCapabilities, getDeviceLimits } from '../device.js';
import type { CommandRecorder } from '../command-recorder.js';
import { getKernelTuner } from '../kernel-tuner.js';

/** Shader source cache (loaded via fetch) */
const shaderSourceCache = new Map<string, string>();

/** Compiled shader module cache */
const shaderModuleCache = new Map<string, Promise<GPUShaderModule>>();

/** Compiled pipeline cache */
const pipelineCache = new Map<string, GPUComputePipeline>();

/** Bind group layout cache */
const bindGroupLayoutCache = new Map<string, GPUBindGroupLayout>();

/** Pipeline layout cache */
const pipelineLayoutCache = new Map<string, GPUPipelineLayout>();

/**
 * Base path for kernel files
 * Detects if running under /doppler/ (replo.id deployment) or standalone
 */
function getKernelBasePath(): string {
  // Check if we're running from /doppler/ path (replo.id deployment)
  if (typeof location !== 'undefined') {
    const path = location.pathname;
    if (path.startsWith('/d') || path.startsWith('/doppler/') || location.host.includes('replo')) {
      return '/doppler/gpu/kernels';
    }
  }
  return '/gpu/kernels';
}

const KERNEL_BASE_PATH = getKernelBasePath();

/** Kernel configuration */
export interface KernelConfig {
  shaderFile: string;
  entryPoint: string;
  workgroupSize: [number, number, number];
  requires: string[];
  validate?: (seqLen: number, numHeads: number, headDim: number) => void;
}

/** All kernel configurations by operation and variant */
export const KERNEL_CONFIGS: Record<string, Record<string, KernelConfig>> = {
  matmul: {
    f16: {
      shaderFile: 'matmul_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [16, 16, 1],
      requires: ['shader-f16'],
    },
    f16_vec4: {
      shaderFile: 'matmul_f16.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [16, 16, 1],
      requires: ['shader-f16'],
    },
    f16w_f32a: {
      shaderFile: 'matmul_f16w_f32a.wgsl',
      entryPoint: 'main',
      workgroupSize: [16, 16, 1],
      requires: ['shader-f16'],
    },
    f16w_f32a_naive: {
      shaderFile: 'matmul_f16w_f32a_naive.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    // Optimized GEMV for M=1 decode: uses shared memory for A vector
    gemv: {
      shaderFile: 'matmul_gemv.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    // Subgroup-optimized GEMV - 1.5x faster using subgroupAdd
    gemv_subgroup: {
      shaderFile: 'matmul_gemv_subgroup.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    gemv_subgroup_vec4: {
      shaderFile: 'matmul_gemv_subgroup.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    // Multi-column GEMV for large vocab (LM head F16) - 32 columns per workgroup
    // Reduces workgroups from 65K to 8K for vocab=262144
    gemv_subgroup_multicol: {
      shaderFile: 'matmul_gemv_subgroup.wgsl',
      entryPoint: 'main_multicol',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    // Fused Q4_K dequant + matmul - 2-3x faster (no separate dequant pass)
    q4_fused: {
      shaderFile: 'matmul_q4_fused.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    q4_fused_batched: {
      shaderFile: 'matmul_q4_fused.wgsl',
      entryPoint: 'main_batched',
      workgroupSize: [64, 4, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    // Multi-column GEMV for large vocab (LM head) - 32 columns per workgroup
    q4_fused_multicol: {
      shaderFile: 'matmul_q4_fused.wgsl',
      entryPoint: 'main_multicol',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    f32: {
      shaderFile: 'matmul_f32.wgsl',
      entryPoint: 'main',
      workgroupSize: [16, 16, 1],
      requires: [],
    },
  },
  // Fused FFN kernels (Tier 2 P0)
  ffn_fused: {
    default: {
      shaderFile: 'ffn_fused.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    multi: {
      shaderFile: 'ffn_fused.wgsl',
      entryPoint: 'main_multi',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    f16: {
      shaderFile: 'ffn_fused.wgsl',
      entryPoint: 'main_f16',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    batched: {
      shaderFile: 'ffn_fused.wgsl',
      entryPoint: 'main_batched',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
  },
  // Optimized attention decode (Tier 2 P0)
  attention_decode_optimized: {
    default: {
      shaderFile: 'attention_decode_optimized.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    multihead: {
      shaderFile: 'attention_decode_optimized.wgsl',
      entryPoint: 'main_multihead',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    f16kv: {
      shaderFile: 'attention_decode_optimized.wgsl',
      entryPoint: 'main_f16kv',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
  },
  dequant: {
    subgroup: {
      shaderFile: 'dequant_subgroup.wgsl',
      entryPoint: 'main',
      workgroupSize: [64, 1, 1],
      requires: ['subgroups'],
    },
    subgroup_vec4: {
      shaderFile: 'dequant_subgroup.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['subgroups'],
    },
    subgroup_f16out: {
      shaderFile: 'dequant_f16_out.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups', 'shader-f16'],
    },
    subgroup_vec4_f16out: {
      shaderFile: 'dequant_f16_out.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['subgroups', 'shader-f16'],
    },
    shared: {
      shaderFile: 'dequant_shared.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    shared_vec4: {
      shaderFile: 'dequant_shared.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    shared_f16out: {
      shaderFile: 'dequant_f16_out.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    shared_vec4_f16out: {
      shaderFile: 'dequant_f16_out.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
    },
    // MXFP4 dequantization (GPT-OSS)
    mxfp4: {
      shaderFile: 'dequant_mxfp4.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    mxfp4_vec4: {
      shaderFile: 'dequant_mxfp4.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    mxfp4_expert: {
      shaderFile: 'dequant_mxfp4.wgsl',
      entryPoint: 'main_expert',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    // Q6_K dequantization (GGUF 6-bit quantization)
    q6k_f16out: {
      shaderFile: 'dequant_q6k.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
  attention: {
    prefill: {
      shaderFile: 'attention.wgsl',
      entryPoint: 'main',
      workgroupSize: [64, 1, 1],
      requires: [],
      validate: validateAttentionLimits,
    },
    decode: {
      shaderFile: 'attention.wgsl',
      entryPoint: 'attention_decode',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    prefill_small: {
      shaderFile: 'attention_small.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: [],
      validate: validateAttentionLimits,
    },
    decode_small: {
      shaderFile: 'attention_small.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: [],
    },
    prefill_streaming: {
      shaderFile: 'attention_streaming.wgsl',
      entryPoint: 'main',
      workgroupSize: [1, 1, 1],
      requires: [],
      validate: validateAttentionLimits,
    },
    decode_streaming: {
      shaderFile: 'attention_streaming.wgsl',
      entryPoint: 'main',
      workgroupSize: [1, 1, 1],
      requires: [],
    },
    prefill_f16kv: {
      shaderFile: 'attention_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
      validate: validateAttentionLimits,
    },
    decode_f16kv: {
      shaderFile: 'attention_f16kv.wgsl',
      entryPoint: 'attention_decode',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    prefill_small_f16kv: {
      shaderFile: 'attention_small_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: ['shader-f16'],
      validate: validateAttentionLimits,
    },
    decode_small_f16kv: {
      shaderFile: 'attention_small_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: ['shader-f16'],
    },
    prefill_streaming_f16kv: {
      shaderFile: 'attention_streaming_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [1, 1, 1],
      requires: ['shader-f16'],
      validate: validateAttentionLimits,
    },
    decode_streaming_f16kv: {
      shaderFile: 'attention_streaming_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [1, 1, 1],
      requires: ['shader-f16'],
    },
    // Chunked decode kernel - parallelizes headDim for models with few heads (e.g., Gemma 3)
    decode_chunked_f16kv: {
      shaderFile: 'attention_decode_chunked_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    // Subgroup-optimized decode kernel - 4 barriers (vs 80), 100% thread utilization
    decode_subgroup: {
      shaderFile: 'attention_decode_subgroup.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],  // headDim threads per workgroup
      requires: ['subgroups'],
    },
  },
  rmsnorm: {
    default: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    small: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'rmsnorm_small',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    residual: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'rmsnorm_inplace_residual',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  softmax: {
    default: {
      shaderFile: 'softmax.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    small: {
      shaderFile: 'softmax.wgsl',
      entryPoint: 'softmax_small',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    online: {
      shaderFile: 'softmax.wgsl',
      entryPoint: 'softmax_online',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  rope: {
    default: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    compute_freqs: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'rope_compute_freqs',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    qk: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'rope_qk',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    ntk: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'rope_ntk_scaled',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    yarn: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'rope_yarn',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  silu: {
    default: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gate: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'silu_gate',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gate_split: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'silu_gate_split',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    vec4: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'silu_vec4',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gelu: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'gelu',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    geglu: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'geglu',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gate_rowsplit: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'silu_gate_rowsplit',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    geglu_rowsplit: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'geglu_rowsplit',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  scale: {
    default: {
      shaderFile: 'scale.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    inplace: {
      shaderFile: 'scale.wgsl',
      entryPoint: 'main_inplace',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  gather: {
    default: {
      shaderFile: 'gather.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    vec4: {
      shaderFile: 'gather.wgsl',
      entryPoint: 'gather_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    // F16 embeddings → F32 output (for weight-tied lm_head optimization)
    f16: {
      shaderFile: 'gather_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    f16_vec4: {
      shaderFile: 'gather_f16.wgsl',
      entryPoint: 'gather_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
    },
  },
  residual: {
    default: {
      shaderFile: 'residual.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    vec4: {
      shaderFile: 'residual.wgsl',
      entryPoint: 'add_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
  },
  topk: {
    default: {
      shaderFile: 'topk.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: [],
    },
    small: {
      shaderFile: 'topk.wgsl',
      entryPoint: 'topk_2_small',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    fused: {
      shaderFile: 'topk.wgsl',
      entryPoint: 'softmax_topk',
      workgroupSize: [32, 1, 1],
      requires: [],
    },
  },
  scatter_add: {
    default: {
      shaderFile: 'scatter_add.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    vec4: {
      shaderFile: 'scatter_add.wgsl',
      entryPoint: 'scatter_add_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    dynamic: {
      shaderFile: 'scatter_add.wgsl',
      entryPoint: 'scatter_add_dynamic',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    accumulate: {
      shaderFile: 'scatter_add.wgsl',
      entryPoint: 'scatter_add_accumulate',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  moe_gather: {
    count: {
      shaderFile: 'moe_gather.wgsl',
      entryPoint: 'count_and_map',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gather: {
      shaderFile: 'moe_gather.wgsl',
      entryPoint: 'gather_tokens',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gather_vec4: {
      shaderFile: 'moe_gather.wgsl',
      entryPoint: 'gather_tokens_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    single_pass: {
      shaderFile: 'moe_gather.wgsl',
      entryPoint: 'gather_single_pass',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  swiglu: {
    rowsplit_bias: {
      shaderFile: 'swiglu.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  bias_add: {
    default: {
      shaderFile: 'bias_add.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  cast: {
    f32_to_f16: {
      shaderFile: 'cast_f32_to_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
  sample: {
    argmax: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'argmax',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    argmax_reduce: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'argmax_reduce',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    find_topk_phase1: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'find_topk_phase1',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    find_topk_phase2: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'find_topk_phase2',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    softmax_and_sample: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'softmax_and_sample',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    single_pass: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'sample_single_pass',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  bf16_to_f32: {
    default: {
      shaderFile: 'bf16_to_f32.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  bf16_to_f16: {
    default: {
      shaderFile: 'bf16_to_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
};

/**
 * Validate that attention parameters are within device limits
 */
export function validateAttentionLimits(seqLen: number, numHeads: number, headDim: number): void {
  const limits = getDeviceLimits();
  if (!limits) return; // No device, validation will fail later

  // Check workgroup invocations limit
  const workgroupInvocations = seqLen * numHeads;
  if (workgroupInvocations > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention parameters exceed device limits: ${workgroupInvocations} workgroups ` +
      `> ${limits.maxComputeWorkgroupsPerDimension} max per dimension. ` +
      `Try reducing seqLen (${seqLen}) or numHeads (${numHeads}).`
    );
  }

  // Check buffer size limits for KV cache
  const kvCacheSize = seqLen * numHeads * headDim * 4; // float32
  if (kvCacheSize > limits.maxStorageBufferBindingSize) {
    throw new Error(
      `KV cache size ${(kvCacheSize / 1e9).toFixed(2)}GB exceeds device limit ` +
      `${(limits.maxStorageBufferBindingSize / 1e9).toFixed(2)}GB. ` +
      `Reduce sequence length or use paged attention.`
    );
  }

  // Check shared memory requirements for attention tile
  const tileSize = 64; // TILE_SIZE in attention.wgsl
  const sharedMemRequired = tileSize * headDim * 4 * 2; // K and V tiles
  if (sharedMemRequired > limits.maxComputeWorkgroupStorageSize) {
    console.warn(
      `[KernelSelector] Attention may be slow: tile requires ${sharedMemRequired} bytes ` +
      `but device has ${limits.maxComputeWorkgroupStorageSize} bytes shared memory.`
    );
  }
}

/**
 * Load a WGSL shader file via fetch
 */
export async function loadShaderSource(filename: string): Promise<string> {
  if (shaderSourceCache.has(filename)) {
    return shaderSourceCache.get(filename)!;
  }

  const url = `${KERNEL_BASE_PATH}/${filename}`;
  try {
    const response = await fetch(url, { cache: 'no-cache' });
    if (!response.ok) {
      throw new Error(`Failed to load shader ${filename}: ${response.status}`);
    }
    const source = await response.text();
    shaderSourceCache.set(filename, source);
    return source;
  } catch (error) {
    console.error(`[KernelSelector] Failed to load shader ${filename}:`, error);
    throw error;
  }
}

/** Minimum capabilities interface for feature checking */
interface FeatureCapabilities {
  hasF16: boolean;
  hasSubgroups: boolean;
}

/**
 * Check if all required features are available
 */
export function hasRequiredFeatures(required: string[], capabilities: FeatureCapabilities): boolean {
  for (const feature of required) {
    if (feature === 'shader-f16' && !capabilities.hasF16) return false;
    if (feature === 'subgroups' && !capabilities.hasSubgroups) return false;
    if (feature === 'subgroups-f16' && !capabilities.hasSubgroups) return false;
  }
  return true;
}

/**
 * Get kernel configuration
 */
export function getKernelConfig(operation: string, variant: string): KernelConfig {
  const config = KERNEL_CONFIGS[operation]?.[variant];
  if (!config) {
    throw new Error(`Unknown kernel: ${operation}/${variant}`);
  }
  return config;
}

/**
 * Compile a shader module
 */
export async function compileShader(device: GPUDevice, source: string, label: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({
    label,
    code: source,
  });

  // Check for compilation errors
  const compilationInfo = await module.getCompilationInfo();
  if (compilationInfo.messages.length > 0) {
    for (const msg of compilationInfo.messages) {
      const type = msg.type === 'error' ? 'ERROR' : msg.type === 'warning' ? 'WARN' : 'INFO';
      console.log(`[DEBUG compileShader ${label}] ${type}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
    }
    if (compilationInfo.messages.some(m => m.type === 'error')) {
      throw new Error(`Shader compilation failed for ${label}`);
    }
  }

  return module;
}

/**
 * Get or create a cached shader module for a shader file.
 */
async function getShaderModule(
  device: GPUDevice,
  shaderFile: string,
  label: string
): Promise<GPUShaderModule> {
  const cacheKey = shaderFile;
  const cached = shaderModuleCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const compilePromise = (async () => {
    const shaderSource = await loadShaderSource(shaderFile);
    return compileShader(device, shaderSource, label);
  })();

  shaderModuleCache.set(cacheKey, compilePromise);

  try {
    return await compilePromise;
  } catch (err) {
    shaderModuleCache.delete(cacheKey);
    throw err;
  }
}

/**
 * Get or create a cached bind group layout.
 */
export function getOrCreateBindGroupLayout(
  label: string,
  entries: GPUBindGroupLayoutEntry[],
  deviceOverride: GPUDevice | null = null
): GPUBindGroupLayout {
  const cached = bindGroupLayoutCache.get(label);
  if (cached) {
    return cached;
  }

  const device = deviceOverride || getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }

  const layout = device.createBindGroupLayout({ label, entries });
  bindGroupLayoutCache.set(label, layout);
  return layout;
}

/**
 * Get or create a cached pipeline layout.
 */
export function getOrCreatePipelineLayout(
  label: string,
  bindGroupLayouts: GPUBindGroupLayout[],
  deviceOverride: GPUDevice | null = null
): GPUPipelineLayout {
  const cached = pipelineLayoutCache.get(label);
  if (cached) {
    return cached;
  }

  const device = deviceOverride || getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }

  const layout = device.createPipelineLayout({
    label,
    bindGroupLayouts,
  });

  pipelineLayoutCache.set(label, layout);
  return layout;
}

/**
 * Create a compute pipeline for a kernel
 */
export async function createPipeline(
  operation: string,
  variant: string,
  bindGroupLayout: GPUBindGroupLayout | null = null
): Promise<GPUComputePipeline> {
  const cacheKey = `${operation}:${variant}`;

  // Return cached pipeline if available
  if (pipelineCache.has(cacheKey)) {
    return pipelineCache.get(cacheKey)!;
  }

  const device = getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }

  const config = getKernelConfig(operation, variant);
  const capabilities = getKernelCapabilities();

  // Verify requirements
  if (!hasRequiredFeatures(config.requires, capabilities)) {
    throw new Error(
      `Kernel ${operation}/${variant} requires features: ${config.requires.join(', ')}`
    );
  }

  // Compile or reuse shader module
  const shaderModule = await getShaderModule(device, config.shaderFile, `${operation}_${variant}`);

  // Create pipeline
  const layoutLabel = bindGroupLayout?.label || `${operation}_${variant}_layout`;
  const pipelineDescriptor: GPUComputePipelineDescriptor = {
    label: `${operation}_${variant}_pipeline`,
    layout: bindGroupLayout
      ? getOrCreatePipelineLayout(layoutLabel, [bindGroupLayout], device)
      : 'auto',
    compute: {
      module: shaderModule,
      entryPoint: config.entryPoint,
    },
  };

  const pipeline = await device.createComputePipelineAsync(pipelineDescriptor);
  pipelineCache.set(cacheKey, pipeline);

  return pipeline;
}

/**
 * Clear the pipeline cache
 */
export function clearKernelCaches(): void {
  pipelineCache.clear();
  shaderSourceCache.clear();
  shaderModuleCache.clear();
  bindGroupLayoutCache.clear();
  pipelineLayoutCache.clear();
}

export function clearPipelineCache(): void {
  clearKernelCaches();
}

/**
 * Get cache statistics
 */
export function getCacheStats(): {
  pipelines: number;
  shaders: number;
  shaderModules: number;
  bindGroupLayouts: number;
  pipelineLayouts: number;
} {
  return {
    pipelines: pipelineCache.size,
    shaders: shaderSourceCache.size,
    shaderModules: shaderModuleCache.size,
    bindGroupLayouts: bindGroupLayoutCache.size,
    pipelineLayouts: pipelineLayoutCache.size,
  };
}

/**
 * Get tuned workgroup size for an operation
 */
export async function getTunedWorkgroupSize(
  operation: string,
  inputSizes: Record<string, number> = {}
): Promise<[number, number, number]> {
  try {
    const tuner = await getKernelTuner();
    const result = tuner.getCachedResult(operation, inputSizes);

    if (result) {
      return result.optimalWorkgroupSize;
    }

    // Run tuning if not cached
    const tuneResult = await tuner.tuneKernel(operation, inputSizes);
    return tuneResult.optimalWorkgroupSize;
  } catch (e: any) {
    console.warn(`[KernelSelector] Tuning failed for ${operation}, using defaults:`, e.message);
    // Return defaults based on operation
    switch (operation) {
      case 'matmul':
        return [16, 16, 1];
      case 'attention':
      case 'rmsnorm':
      case 'softmax':
        return [256, 1, 1];
      case 'dequant':
        return [64, 1, 1];
      default:
        return [256, 1, 1];
    }
  }
}

/**
 * Run auto-tuning for all kernels with given model config
 */
export async function autoTuneKernels(modelConfig: Record<string, number> = {}): Promise<Record<string, any>> {
  const {
    hiddenSize = 4096,
    intermediateSize = 14336,
    numHeads = 32,
    headDim = 128,
    maxSeqLen = 4096,
    vocabSize = 32000,
  } = modelConfig;

  const tuner = await getKernelTuner();
  const results: Record<string, any> = {};

  // Tune matmul for common sizes
  results.matmul_hidden = await tuner.tuneKernel('matmul', {
    M: 1, N: hiddenSize, K: hiddenSize,
  });
  results.matmul_ffn = await tuner.tuneKernel('matmul', {
    M: 1, N: intermediateSize, K: hiddenSize,
  });

  // Tune attention
  results.attention = await tuner.tuneKernel('attention', {
    seqLen: 1, numHeads, headDim,
  });

  // Tune softmax (LM head output)
  results.softmax = await tuner.tuneKernel('softmax', {
    innerSize: vocabSize, outerSize: 1,
  });

  // Tune RMSNorm
  results.rmsnorm = await tuner.tuneKernel('rmsnorm', {
    hiddenSize, numTokens: 1,
  });

  // Tune dequant
  results.dequant = await tuner.tuneKernel('dequant', {
    numBlocks: 1000,
  });

  console.log('[KernelSelector] Auto-tuning complete:', results);
  return results;
}

/**
 * Prewarm all supported kernel pipelines
 */
export async function prewarmKernels(
  options: { mode?: 'parallel' | 'sequential' } = {}
): Promise<void> {
  const caps = getKernelCapabilities();
  const mode = options.mode ?? 'parallel';
  const entries = Object.entries(KERNEL_CONFIGS)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([operation, variants]) => [operation, Object.entries(variants).sort(([a], [b]) => a.localeCompare(b))] as const);

  if (mode === 'sequential') {
    let count = 0;
    for (const [operation, variants] of entries) {
      for (const [variant, cfg] of variants) {
        if (cfg.requires && !hasRequiredFeatures(cfg.requires, caps)) {
          continue;
        }
        try {
          await createPipeline(operation, variant);
          count += 1;
        } catch (e: any) {
          console.warn(`[KernelSelector] Prewarm failed for ${operation}/${variant}:`, e.message);
        }
      }
    }
    console.log(`[KernelSelector] Prewarmed ${count} kernel pipelines`);
    return;
  }

  const jobs: Promise<void>[] = [];
  for (const [operation, variants] of entries) {
    for (const [variant, cfg] of variants) {
      if (cfg.requires && !hasRequiredFeatures(cfg.requires, caps)) {
        continue;
      }
      jobs.push(
        createPipeline(operation, variant).catch((e) => {
          console.warn(`[KernelSelector] Prewarm failed for ${operation}/${variant}:`, e.message);
        })
      );
    }
  }

  await Promise.all(jobs);
  console.log(`[KernelSelector] Prewarmed ${jobs.length} kernel pipelines`);
}

export function createUniformBufferFromData(
  label: string,
  data: ArrayBuffer | ArrayBufferView,
  recorder?: CommandRecorder | null,
  deviceOverride?: GPUDevice | null
): GPUBuffer {
  if (recorder) {
    return recorder.createUniformBuffer(data, label);
  }

  const device = deviceOverride ?? getDevice();
  if (!device) {
    throw new Error('GPU device not initialized');
  }

  const byteLength = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
  const buffer = device.createBuffer({
    label,
    size: byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data as GPUAllowSharedBufferSource);
  return buffer;
}

export function createUniformBufferWithView(
  label: string,
  byteLength: number,
  writer: (view: DataView) => void,
  recorder?: CommandRecorder | null,
  deviceOverride?: GPUDevice | null
): GPUBuffer {
  const data = new ArrayBuffer(byteLength);
  const view = new DataView(data);
  writer(view);
  return createUniformBufferFromData(label, data, recorder, deviceOverride);
}
