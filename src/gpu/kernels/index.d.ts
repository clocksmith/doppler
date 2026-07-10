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
  isFusedQ4KDisabled,
  type MatmulOptions,
} from './matmul.js';
export {
  recordLmHeadArgmax,
  recordLmHeadArgmaxF16,
  type LmHeadArgmaxOptions,
  type LmHeadArgmaxF16Options,
} from './lm-head-argmax.js';
export {
  runLmHeadSelectLogitsF16,
  type LmHeadSelectLogitsF16Options,
  type LmHeadSelectLogitsF16Result,
} from './lm-head-select-logits.js';

// Dequantization
export {
  selectDequantKernel,
  createDequantBindGroupLayout,
  dequantize,
  dequantizeRowwise,
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
  runAttentionTiered,
  recordAttentionTiered,
  runAttentionTieredQuant,
  recordAttentionTieredQuant,
  runAttentionBDPA,
  recordAttentionBDPA,
  type AttentionOptions,
  type TieredAttentionOptions,
  type TieredQuantAttentionOptions,
  type BDPAAttentionOptions,
} from './attention.js';

// RMSNorm
export {
  selectRMSNormKernel,
  runRMSNorm,
  recordRMSNorm,
  type RMSNormOptions,
} from './rmsnorm.js';
export {
  canUseRMSNormQK,
  runRMSNormQK,
  recordRMSNormQK,
  type RMSNormQKOptions,
  type RMSNormQKResult,
} from './rmsnorm-qk.js';
export {
  canUseSplitQKVRMSNormQK,
  runSplitQKVRMSNormQK,
  recordSplitQKVRMSNormQK,
  type SplitQKVRMSNormQKOptions,
  type SplitQKVRMSNormQKResult,
} from './split-qkv-rmsnorm-qk.js';
export {
  canUseSplitQKVRMSNormRoPEQK,
  runSplitQKVRMSNormRoPEQK,
  recordSplitQKVRMSNormRoPEQK,
  type SplitQKVRMSNormRoPEQKOptions,
  type SplitQKVRMSNormRoPEQKResult,
} from './split-qkv-rmsnorm-rope-qk.js';
export {
  RMSNORM_PAIR_CACHE_LIMIT,
  runSandwichRMSNormPair,
  recordSandwichRMSNormPair,
  runResidualNextRMSNormPair,
  recordResidualNextRMSNormPair,
  type SandwichRMSNormPairOptions,
  type SandwichRMSNormPairResult,
  type ResidualNextRMSNormPairOptions,
  type ResidualNextRMSNormPairResult,
} from './rmsnorm-pair.js';

// LayerNorm
export {
  selectLayerNormKernel,
  runLayerNorm,
  recordLayerNorm,
  type LayerNormOptions,
} from './layernorm.js';

// Softmax
export {
  runSoftmax,
  runSoftmaxTopK,
  recordSoftmax,
  type SoftmaxOptions,
} from './softmax.js';

export {
  runSoftEmbeddingSplitF16,
  runSoftEmbeddingLogitsF16,
  type SoftEmbeddingSplitOptions,
  type SoftEmbeddingLogitsOptions,
} from './soft-embedding.js';

export {
  runDiffusionGemmaCanvasStats,
  type DiffusionGemmaCanvasStatsOptions,
  type DiffusionGemmaCanvasStatsResult,
} from './diffusion-gemma-sampling.js';

export {
  runGemma4RouteQ4MatmulF16A,
  runScatterAddRoutesF16ExpertScale,
  type Gemma4RouteQ4MatmulOptions,
  type ScatterAddRoutesF16ExpertScaleOptions,
} from './gemma4-route-expert.js';

export {
  runKVQuantize,
  recordKVQuantize,
  type KVQuantizeOptions,
} from './kv-quantize.js';

export {
  runKVCacheWriteF32ToF16,
  recordKVCacheWriteF32ToF16,
  type KVCacheWriteF32ToF16Options,
} from './kv-cache-write.js';

// Loss
export {
  runCrossEntropyLoss,
  recordCrossEntropyLoss,
  type CrossEntropyLossOptions,
} from './cross_entropy_loss.js';

// RoPE
export {
  runRoPE,
  recordRoPE,
  type RoPEOptions,
} from './rope.js';
export {
  canUseRoPEQK,
  runRoPEQK,
  recordRoPEQK,
  type RoPEQKOptions,
  type RoPEQKResult,
} from './rope-qk.js';

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
  selectScaleKernel,
  runScale,
  recordScale,
  type ScaleOptions,
} from './scale.js';

// Clamp
export {
  runClamp,
  recordClamp,
  type ClampOptions,
} from './clamp.js';

// Static activation quantize/dequantize
export {
  runActivationStaticQdq,
  recordActivationStaticQdq,
  type ActivationStaticQdqOptions,
} from './activation-static-qdq.js';

// Energy (EBM helpers)
export {
  runEnergyEval,
  recordEnergyEval,
  runEnergyUpdate,
  recordEnergyUpdate,
  runEnergyQuintelUpdate,
  recordEnergyQuintelUpdate,
  runEnergyQuintelReduce,
  recordEnergyQuintelReduce,
  runEnergyQuintelGrad,
  recordEnergyQuintelGrad,
  type EnergyEvalOptions,
  type EnergyUpdateOptions,
  type EnergyQuintelUpdateOptions,
  type EnergyQuintelReduceOptions,
  type EnergyQuintelGradOptions,
} from './energy.js';

// Conv2D
export {
  runConv2D,
  recordConv2D,
  type Conv2DOptions,
} from './conv2d.js';

export {
  runDepthwiseConv2D,
  recordDepthwiseConv2D,
  type DepthwiseConv2DOptions,
} from './depthwise_conv2d.js';

export {
  runDepthwiseConv1D,
  recordDepthwiseConv1D,
  type DepthwiseConv1DOptions,
} from './depthwise_conv1d.js';

export {
  runGroupedPointwiseConv2D,
  recordGroupedPointwiseConv2D,
  type GroupedPointwiseConv2DOptions,
} from './grouped_pointwise_conv2d.js';

// Gather (Embedding Lookup)
export {
  runGather,
  recordGather,
  runGatherSplit4,
  recordGatherSplit4,
  runGatherSplit8,
  recordGatherSplit8,
  runGatherSplit,
  recordGatherSplit,
  type GatherOptions,
} from './gather.js';

// GroupNorm
export {
  runGroupNorm,
  recordGroupNorm,
  type GroupNormOptions,
} from './groupnorm.js';

// Modulate
export {
  runModulate,
  recordModulate,
  type ModulateOptions,
} from './modulate.js';

// Residual Connections
export {
  runResidualAdd,
  runBiasAdd,
  recordResidualAdd,
  recordBiasAdd,
  type ResidualOptions,
} from './residual.js';

// Pixel Shuffle
export {
  runPixelShuffle,
  recordPixelShuffle,
  type PixelShuffleOptions,
} from './pixel_shuffle.js';

// Upsample2D
export {
  runUpsample2D,
  recordUpsample2D,
  type Upsample2DOptions,
} from './upsample2d.js';

// Mixture of Experts
export {
  runTopK,
  runMoEGather,
  runMoEBuildTokenOffsets,
  recordMoEBuildTokenOffsets,
  runScatterAdd,
  runScatterAddDynamic,
  type MoEOptions,
} from './moe.js';

// Type Casting
export {
  castF32ToF16,
  recordCastF32ToF16,
  castF16ToF32,
  recordCastF16ToF32,
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

export {
  runLinearAttention,
  recordLinearAttention,
  type LinearAttentionOptions,
} from './linear_attention.js';

export {
  runRepeatChannels,
  recordRepeatChannels,
  type RepeatChannelsOptions,
} from './repeat_channels.js';

export {
  runReLU,
  recordReLU,
  type ReLUOptions,
} from './relu.js';

// Fused FFN (Tier 2 P0)
export {
  runFusedFFN,
  recordFusedFFN,
  runFusedFFNFromRMSNormStats,
  recordFusedFFNFromRMSNormStats,
  calculateFusedFFNSavings,
  type FusedFFNOptions,
  type FusedNormedFFNOptions,
  type FFNActivation,
} from './fused_ffn.js';

export {
  runRMSNormStats,
  recordRMSNormStats,
  type RMSNormStatsOptions,
  type RMSNormStatsResult,
} from './rmsnorm-stats.js';

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

// Split QKV
export {
  runSplitQKV,
  recordSplitQKV,
  type SplitQKVOptions,
  type SplitQKVResult,
} from './split_qkv.js';

// Split Q and Gate (de-interleave attentionOutputGate q_proj output)
export {
  runSplitQG,
  recordSplitQG,
  type SplitQGOptions,
  type SplitQGResult,
} from './split_qg.js';

// Transpose
export {
  runTranspose,
  recordTranspose,
  type TransposeOptions,
} from './transpose.js';

// Training Backward Kernels
export {
  runEmbedBackward,
  recordEmbedBackward,
  runBiasAddBackward,
  runUpsample2DBackward,
  runGroupNormBackward,
  runConv2DBackward,
  runMatmulBackward,
  recordMatmulBackward,
  runSoftmaxBackward,
  recordSoftmaxBackward,
  runRmsNormBackward,
  recordRmsNormBackward,
  runLayerNormBackward,
  recordLayerNormBackward,
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
