/**
 * GPU Kernels - Barrel Export
 *
 * Central export point for all GPU kernel modules.
 * This allows backward compatibility with the original kernel-selector.js
 */

// Utilities
export {
  KERNEL_CONFIGS,
  validateAttentionLimits,
  loadShaderSource,
  hasRequiredFeatures,
  getKernelConfig,
  compileShader,
  getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout,
  createPipeline,
  clearKernelCaches,
  clearPipelineCache,
  getCacheStats,
  getTunedWorkgroupSize,
  autoTuneKernels,
  prewarmKernels,
  type KernelConfig,
} from './utils.js';

export type {
  OutputBufferOptions,
  OutputOffsetOptions,
  OutputDtypeOptions,
  Vec4Options,
} from './types.js';

// Matrix Multiplication
export {
  selectMatmulKernel,
  createMatmulBindGroupLayout,
  runMatmul,
  recordMatmul,
  type MatmulOptions,
} from './matmul.js';

// Dequantization
export {
  selectDequantKernel,
  createDequantBindGroupLayout,
  dequantize,
  dequantizeQ6K,
  dequantizeMXFP4,
  dequantizeMXFP4Expert,
  recordDequantize,
  type DequantOptions,
} from './dequant.js';

// Attention
export {
  runAttention,
  recordAttention,
  type AttentionOptions,
} from './attention.js';

// RMSNorm
export {
  selectRMSNormKernel,
  runRMSNorm,
  recordRMSNorm,
  type RMSNormOptions,
} from './rmsnorm.js';

// Softmax
export {
  runSoftmax,
  runSoftmaxTopK,
  recordSoftmax,
  type SoftmaxOptions,
} from './softmax.js';

// RoPE
export {
  runRoPE,
  recordRoPE,
  type RoPEOptions,
} from './rope.js';

// SiLU Activation
export {
  runSiLU,
  runSwiGLURowsplitBias,
  runSiLURowSplit,
  recordSiLU,
  recordSiLURowSplit,
  type SiLUOptions,
  type SiLURowSplitOptions,
} from './silu.js';

// GeLU Activation
export {
  runGeLU,
  recordGeLU,
  type GeLUOptions,
} from './gelu.js';

// Gather (Embedding Lookup)
export {
  runGather,
  recordGather,
  type GatherOptions,
} from './gather.js';

// Residual Connections
export {
  runResidualAdd,
  runBiasAdd,
  recordResidualAdd,
  recordBiasAdd,
  type ResidualOptions,
} from './residual.js';

// Mixture of Experts
export {
  runTopK,
  runMoEGather,
  runScatterAdd,
  runScatterAddDynamic,
  type MoEOptions,
} from './moe.js';

// Type Casting
export {
  castF32ToF16,
  recordCastF32ToF16,
  runBF16ToF32,
  runBF16ToF16,
  type CastOptions,
} from './cast.js';

// GPU-Side Sampling
export {
  runArgmax,
  runGPUSample,
  recordArgmax,
  isGPUSamplingAvailable,
  type SampleOptions,
  type SampleResult,
} from './sample.js';

// Re-export CommandRecorder types for convenience
export {
  CommandRecorder,
  createCommandRecorder,
  createProfilingRecorder,
  type RecorderOptions,
  type ProfileTimings,
} from '../command-recorder.js';
