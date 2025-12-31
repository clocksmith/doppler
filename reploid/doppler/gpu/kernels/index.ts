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
  dequantizeQ8_0,
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

// Scale (Element-wise Multiply by Scalar)
export {
  runScale,
  recordScale,
  type ScaleOptions,
} from './scale.js';

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

// Fused FFN (Tier 2 P0)
export {
  runFusedFFN,
  recordFusedFFN,
  calculateFusedFFNSavings,
  type FusedFFNOptions,
  type FFNActivation,
} from './fused_ffn.js';

// Fused Matmul + RMSNorm (P0 - 1.2-1.5x decode speedup)
export {
  selectMatmulRMSNormFusedVariant,
  runMatmulRMSNormFused,
  recordMatmulRMSNormFused,
  shouldUseFusedMatmulRMSNorm,
  type MatmulRMSNormFusedOptions,
} from './fused_matmul_rmsnorm.js';

// Re-export for convenience in layer.ts integration
export { recordMatmulRMSNormFused as doRecordMatmulRMSNormFused } from './fused_matmul_rmsnorm.js';

// Fused Matmul + Residual (P1 - eliminates 1 dispatch per layer for attention output)
export {
  runMatmulResidualFused,
  recordMatmulResidualFused,
  shouldUseFusedMatmulResidual,
  type MatmulResidualFusedOptions,
} from './fused_matmul_residual.js';

// Re-export CommandRecorder types for convenience
export {
  CommandRecorder,
  createCommandRecorder,
  createProfilingRecorder,
  type RecorderOptions,
  type ProfileTimings,
} from '../command-recorder.js';

// Re-export benchmark utilities
export {
  benchmarkMatmul,
  benchmarkAttentionDecode,
  benchmarkRMSNorm,
  benchmarkSiLU,
  benchmarkMatmulRMSNormFused,
  benchmarkDecodePass,
  compareBenchmarks,
  exportBenchmarkJSON,
  printBenchmarkReport,
  type KernelBenchmarkResult,
  type BenchmarkComparison,
  type BenchmarkReport,
  type BenchmarkConfig,
} from '../kernel-benchmark.js';

// Split QKV
export {
  runSplitQKV,
  recordSplitQKV,
  type SplitQKVOptions,
  type SplitQKVResult,
} from './split_qkv.js';

// Re-export profiling utilities
export {
  isProfilingEnabled,
  setProfilingEnabled,
  clearProfile,
  startProfileSession,
  recordProfileEntry,
  profileAsync,
  profileSync,
  profileKernel,
  getProfileReport,
  printProfileReport,
  exportProfileJSON,
  analyzeDecodePerformance,
  type ProfileEntry,
  type ProfileReport,
} from '../perf-profiler.js';
