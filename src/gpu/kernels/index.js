

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
} from './utils.js';

// Matrix Multiplication
export {
  selectMatmulKernel,
  createMatmulBindGroupLayout,
  runMatmul,
  recordMatmul,
  isFusedQ4KDisabled,
} from './matmul.js';

// Dequantization
export {
  selectDequantKernel,
  createDequantBindGroupLayout,
  dequantize,
  dequantizeRowwise,
  dequantizeQ6K,
  dequantizeQ8_0,
  dequantizeMXFP4,
  dequantizeMXFP4Expert,
  recordDequantize,
} from './dequant.js';

// Attention
export {
  runAttention,
  recordAttention,
  runAttentionTiered,
  recordAttentionTiered,
  runAttentionTieredQuant,
  recordAttentionTieredQuant,
} from './attention.js';

// RMSNorm
export {
  selectRMSNormKernel,
  runRMSNorm,
  recordRMSNorm,
} from './rmsnorm.js';

// Softmax
export {
  runSoftmax,
  runSoftmaxTopK,
  recordSoftmax,
} from './softmax.js';

// KV Quantization
export {
  runKVQuantize,
  recordKVQuantize,
} from './kv-quantize.js';

// Loss
export {
  runCrossEntropyLoss,
  recordCrossEntropyLoss,
} from './cross_entropy_loss.js';

// RoPE
export {
  runRoPE,
  recordRoPE,
} from './rope.js';

// SiLU Activation
export {
  runSiLU,
  runSwiGLURowsplitBias,
  runSiLURowSplit,
  recordSiLU,
  recordSiLURowSplit,
} from './silu.js';

// GeLU Activation
export {
  runGeLU,
  recordGeLU,
} from './gelu.js';

// Scale (Element-wise Multiply by Scalar)
export {
  runScale,
  recordScale,
} from './scale.js';

// Gather (Embedding Lookup)
export {
  runGather,
  recordGather,
} from './gather.js';

// Residual Connections
export {
  runResidualAdd,
  runBiasAdd,
  recordResidualAdd,
  recordBiasAdd,
} from './residual.js';

// Mixture of Experts
export {
  runTopK,
  runMoEGather,
  runScatterAdd,
  runScatterAddDynamic,
} from './moe.js';

// Type Casting
export {
  castF32ToF16,
  recordCastF32ToF16,
  castF16ToF32,
  recordCastF16ToF32,
  runBF16ToF32,
  runBF16ToF16,
} from './cast.js';

// GPU-Side Sampling
export {
  runArgmax,
  runGPUSample,
  recordArgmax,
  isGPUSamplingAvailable,
} from './sample.js';

// Fused FFN (Tier 2 P0)
export {
  runFusedFFN,
  recordFusedFFN,
  calculateFusedFFNSavings,
} from './fused_ffn.js';

// Fused Matmul + RMSNorm (P0 - 1.2-1.5x decode speedup)
export {
  selectMatmulRMSNormFusedVariant,
  runMatmulRMSNormFused,
  recordMatmulRMSNormFused,
  shouldUseFusedMatmulRMSNorm,
} from './fused_matmul_rmsnorm.js';

// Re-export for convenience in layer.ts integration
export { recordMatmulRMSNormFused as doRecordMatmulRMSNormFused } from './fused_matmul_rmsnorm.js';

// Fused Matmul + Residual (P1 - eliminates 1 dispatch per layer for attention output)
export {
  runMatmulResidualFused,
  recordMatmulResidualFused,
  shouldUseFusedMatmulResidual,
} from './fused_matmul_residual.js';

// Re-export CommandRecorder types for convenience
export {
  CommandRecorder,
  createCommandRecorder,
  createProfilingRecorder,
} from '../command-recorder.js';

// Split QKV
export {
  runSplitQKV,
  recordSplitQKV,
} from './split_qkv.js';

// Transpose
export {
  runTranspose,
  recordTranspose,
} from './transpose.js';

// Training Backward Kernels
export {
  runEmbedBackward,
  recordEmbedBackward,
  runMatmulBackward,
  recordMatmulBackward,
  runSoftmaxBackward,
  recordSoftmaxBackward,
  runRmsNormBackward,
  recordRmsNormBackward,
  runAttentionBackward,
  recordAttentionBackward,
  runRoPEBackward,
  recordRoPEBackward,
  runSiluBackward,
  recordSiluBackward,
  runGeluBackward,
  recordGeluBackward,
  runScaleBackward,
  recordScaleBackward,
  runCrossEntropyBackward,
  recordCrossEntropyBackward,
  runAdam,
  recordAdam,
} from './backward/index.js';

// Re-export profiling utilities
