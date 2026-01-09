import { isVariantAvailable } from './registry.js';
import {
  getCapabilities,
  getPreferredVariant,
  shouldAvoidVariant,
} from '../platforms/loader.js';

// =============================================================================
// Matmul Selection
// =============================================================================


export function selectMatmul(context) {
  const {
    M,
    N,
    K,
    aDtype = 'f32',
    bDtype = 'f32',
    outputDtype = 'f32',
    preferF16 = true,
    useVec4 = false,
  } = context;

  let capabilities;
  try {
    capabilities = getCapabilities();
  } catch {
    // Fallback if platform not initialized
    capabilities = { hasF16: false, hasSubgroups: false };
  }

  // Check for platform preference
  try {
    const preferred = getPreferredVariant('matmul');
    if (preferred && isVariantAvailable('matmul', preferred, capabilities)) {
      if (!shouldAvoidVariant('matmul', preferred)) {
        return preferred;
      }
    }
  } catch {
    // Platform not initialized, continue with default logic
  }

  const isDecode = M === 1;
  const inputsAreF16 = aDtype === 'f16' && bDtype === 'f16';
  const weightsAreF16 = bDtype === 'f16' && aDtype !== 'f16';

  // Q4K fused path - 2-3x faster than separate dequant + matmul
  if (bDtype === 'q4k' && capabilities.hasSubgroups) {
    if (N > 8192 && isDecode) {
      // Multi-column for large vocab (LM head)
      return 'q4_fused_multicol';
    }
    return isDecode ? 'q4_fused' : 'q4_fused_batched';
  }

  // Full F16 matmul when both inputs are F16 and output is F16
  if (outputDtype === 'f16' && preferF16 && inputsAreF16 && capabilities.hasF16) {
    return useVec4 ? 'f16_vec4' : 'f16';
  }

  // Mixed precision: F32 activations, F16 weights
  if (weightsAreF16 && capabilities.hasF16) {
    return 'f16w_f32a';
  }

  // GEMV path for decode (M=1) with subgroups
  if (isDecode && capabilities.hasSubgroups && capabilities.hasF16) {
    if (N > 8192) {
      // Multi-column for large vocab
      return 'gemv_subgroup_multicol';
    }
    return useVec4 ? 'gemv_subgroup_vec4' : 'gemv_subgroup';
  }

  // Standard GEMV for decode
  if (isDecode && capabilities.hasF16) {
    return 'gemv';
  }

  // F16 tiled matmul for batched
  if (preferF16 && capabilities.hasF16) {
    return useVec4 ? 'f16_vec4' : 'f16';
  }

  // Fallback to F32
  return 'f32';
}

// =============================================================================
// Attention Selection
// =============================================================================


// Dimension limits for attention variants
const ATTENTION_LIMITS = {
  LARGE_MAX_HEAD_DIM: 64,
  SMALL_MAX_HEAD_DIM: 64,
  SUBGROUP_MAX_HEAD_DIM: 256,
  CHUNKED_MAX_HEAD_DIM: 256,
};

// Shared memory requirements
const ATTENTION_SHARED_MEMORY = {
  LARGE: 49152,
  SMALL_F32: 8192,
  SMALL_F16: 4096,
};

export function selectAttention(context) {
  const {
    seqLen,
    kvSeqLen,
    numHeads,
    headDim,
    useF16KV = false,
    sharedMemoryLimit = 32768,
  } = context;

  let capabilities;
  try {
    capabilities = getCapabilities();
  } catch {
    capabilities = { hasF16: false, hasSubgroups: false };
  }

  // Check for platform preference
  try {
    const preferred = getPreferredVariant('attention');
    if (preferred && isVariantAvailable('attention', preferred, capabilities)) {
      if (!shouldAvoidVariant('attention', preferred)) {
        return preferred;
      }
    }
  } catch {
    // Platform not initialized
  }

  const isDecode = seqLen === 1;
  const suffix = useF16KV ? '_f16kv' : '';

  // Subgroup-optimized decode for single token
  if (
    isDecode &&
    capabilities.hasSubgroups &&
    headDim <= ATTENTION_LIMITS.SUBGROUP_MAX_HEAD_DIM
  ) {
    return 'decode_subgroup';
  }

  // Chunked decode for models with few heads (parallelizes headDim)
  if (
    isDecode &&
    capabilities.hasF16 &&
    headDim <= ATTENTION_LIMITS.CHUNKED_MAX_HEAD_DIM
  ) {
    return `decode_chunked${suffix}`;
  }

  // Check shared memory for tiled variants
  const canLarge =
    headDim <= ATTENTION_LIMITS.LARGE_MAX_HEAD_DIM &&
    sharedMemoryLimit >= ATTENTION_SHARED_MEMORY.LARGE;

  const smallRequired = useF16KV
    ? ATTENTION_SHARED_MEMORY.SMALL_F16
    : ATTENTION_SHARED_MEMORY.SMALL_F32;
  const canSmall =
    headDim <= ATTENTION_LIMITS.SMALL_MAX_HEAD_DIM &&
    sharedMemoryLimit >= smallRequired;

  // Prefill path
  if (!isDecode) {
    if (canLarge) {
      return `prefill${suffix}`;
    }
    if (canSmall) {
      return `prefill_small${suffix}`;
    }
    return `prefill_streaming${suffix}`;
  }

  // Decode path
  if (canLarge) {
    return `decode${suffix}`;
  }
  if (canSmall) {
    return `decode_small${suffix}`;
  }
  return `decode_streaming${suffix}`;
}

// =============================================================================
// Dequant Selection
// =============================================================================


export function selectDequant(context) {
  const {
    quantType = 'q4k',
    outputDtype = 'f32',
    useVec4 = true,
    isExpert = false,
  } = context;

  let capabilities;
  try {
    capabilities = getCapabilities();
  } catch {
    capabilities = { hasF16: false, hasSubgroups: false };
  }

  // Special quantization formats
  if (quantType === 'q6k' && capabilities.hasF16) {
    return 'q6k_f16out';
  }
  if (quantType === 'q8_0' && capabilities.hasF16) {
    return 'q8_0_f16out';
  }
  if (quantType === 'mxfp4') {
    if (isExpert) return 'mxfp4_expert';
    return useVec4 ? 'mxfp4_vec4' : 'mxfp4';
  }

  // Standard Q4K dequant
  const wantsF16Out = outputDtype === 'f16' && capabilities.hasF16;

  if (capabilities.hasSubgroups) {
    if (wantsF16Out) {
      return useVec4 ? 'subgroup_vec4_f16out' : 'subgroup_f16out';
    }
    return useVec4 ? 'subgroup_vec4' : 'subgroup';
  }

  if (wantsF16Out) {
    return useVec4 ? 'shared_vec4_f16out' : 'shared_f16out';
  }
  return useVec4 ? 'shared_vec4' : 'shared';
}

// =============================================================================
// RMSNorm Selection
// =============================================================================


export function selectRMSNorm(context) {
  const { hiddenSize = null, hasResidual = false } = context;

  if (hasResidual) {
    return 'residual';
  }
  if (hiddenSize !== null && hiddenSize <= 256) {
    return 'small';
  }
  return 'default';
}

// =============================================================================
// Fused Matmul+RMSNorm Selection
// =============================================================================

const WG_SIZE = 256;
const MAX_MEDIUM_N = WG_SIZE * 16; // 4096


export function selectFusedMatmulRMSNorm(context) {
  const { N } = context;

  if (N <= WG_SIZE) {
    return 'small'; // Single workgroup, one element per thread
  }
  if (N <= MAX_MEDIUM_N) {
    return 'medium'; // Single workgroup, multiple elements per thread
  }
  // For very large N, fall back to default (incomplete RMSNorm)
  return 'default';
}

// =============================================================================
// FFN Selection
// =============================================================================


export function selectFFN(context) {
  const { batchSize, intermediateSize, weightDtype = 'f32' } = context;

  let capabilities;
  try {
    capabilities = getCapabilities();
  } catch {
    capabilities = { hasF16: false, hasSubgroups: false };
  }

  // For small intermediate sizes, use multi-output variant
  if (intermediateSize <= 1024 && batchSize === 1) {
    return 'multi';
  }

  // For batched execution
  if (batchSize > 1) {
    return 'batched';
  }

  // For F16 weights with F16 support
  if (weightDtype === 'f16' && capabilities.hasF16) {
    return 'f16';
  }

  // Default
  return 'default';
}

// =============================================================================
// Softmax Selection
// =============================================================================


export function selectSoftmax(context) {
  const { innerSize } = context;

  if (innerSize <= 256) {
    return 'small';
  }
  // Online softmax is numerically stable for large sizes
  return 'online';
}

// =============================================================================
// Gather (Embedding) Selection
// =============================================================================


export function selectGather(context) {
  const { embeddingDtype = 'f32', useVec4 = true } = context;

  let capabilities;
  try {
    capabilities = getCapabilities();
  } catch {
    capabilities = { hasF16: false, hasSubgroups: false };
  }

  if (embeddingDtype === 'f16' && capabilities.hasF16) {
    return useVec4 ? 'f16_vec4' : 'f16';
  }
  return useVec4 ? 'vec4' : 'default';
}

// =============================================================================
// Sample Selection
// =============================================================================


export function selectSample(context) {
  const { vocabSize, temperature = 0, topK = 0 } = context;

  // Greedy sampling (temperature=0)
  if (temperature === 0) {
    if (vocabSize > 65536) {
      return 'argmax_reduce'; // Two-phase for large vocab
    }
    return 'argmax';
  }

  // Temperature sampling with top-k
  if (topK > 0 && topK < 100) {
    return 'single_pass'; // Efficient for small top-k
  }

  // Full softmax + sampling
  return 'softmax_and_sample';
}

// =============================================================================
// Rope Selection
// =============================================================================


export function selectRope(context) {
  const { ropeType = 'default', computeFreqs = false, applyToQK = true } = context;

  if (computeFreqs) {
    return 'compute_freqs';
  }
  if (applyToQK) {
    return 'qk';
  }
  if (ropeType === 'ntk') {
    return 'ntk';
  }
  if (ropeType === 'yarn') {
    return 'yarn';
  }
  return 'default';
}

// =============================================================================
// SiLU/Activation Selection
// =============================================================================


export function selectActivation(context) {
  const {
    activation = 'silu',
    hasGate = false,
    rowSplit = false,
    useVec4 = false,
  } = context;

  if (activation === 'gelu') {
    return 'gelu';
  }
  if (activation === 'geglu') {
    return rowSplit ? 'geglu_rowsplit' : 'geglu';
  }

  // SiLU variants
  if (hasGate) {
    if (rowSplit) return 'gate_rowsplit';
    return 'gate_split';
  }
  if (useVec4) {
    return 'vec4';
  }
  return 'default';
}

// =============================================================================
// Residual Selection
// =============================================================================

export function selectResidual(context) {
  const { useVec4 = true } = context;
  return useVec4 ? 'vec4' : 'default';
}

// =============================================================================
// Scatter Add Selection
// =============================================================================

export function selectScatterAdd(context) {
  const { useVec4 = true, accumulate = false, dynamic = false } = context;

  if (accumulate) return 'accumulate';
  if (dynamic) return 'dynamic';
  return useVec4 ? 'vec4' : 'default';
}
