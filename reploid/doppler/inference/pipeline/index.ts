/**
 * Pipeline Module Index
 *
 * Re-exports all pipeline sub-modules for easier importing.
 *
 * Usage:
 *   import { sample, softmax, parseModelConfig } from './pipeline/index.js';
 *
 * Or import specific modules:
 *   import { sample } from './pipeline/sampling.js';
 *
 * @module inference/pipeline
 */

// ============================================================================
// Shared Types
// ============================================================================

export type {
  PipelineContext,
  KVCacheInterface,
  TokenizerInterface,
  GenerateOptions as GenerateOptionsBase,
  GenerationResult as GenerationResultBase,
  LayerConfig,
  LayerWeights,
  ExpertWeights,
  RouterWeights,
  PipelineStats as PipelineStatsBase,
  BatchingStats as BatchingStatsBase,
  MaybeGPUBuffer,
  DecodeFunction,
  RoPEOptions,
} from './types.js';

// ============================================================================
// Configuration
// ============================================================================

export {
  parseModelConfig,
  isGemma3Model,
  isLlama3InstructModel,
  isQwen3Model,
  isKimiK2Model,
  isMixtralModel,
  isGptOssModel,
  normalizeActivation,
  getStopTokenIds,
  inferAttentionParams,
  inferVocabSize,
} from './config.js';

export type {
  ActivationType,
  RawConfig,
  RopeScalingConfig,
  TensorInfo,
  Manifest,
  AttentionParams,
  ParsedModelConfig,
} from './config.js';

// ============================================================================
// Sampling
// ============================================================================

export {
  sample,
  softmax,
  applyRepetitionPenalty,
  getTopK,
  logitsSanity,
} from './sampling.js';

export type {
  SamplingOptions,
  TokenCandidate,
  TopKResult,
  LogitStats,
} from './sampling.js';

// ============================================================================
// Embedding
// ============================================================================

export {
  embed,
  scaleGPUBuffer,
  validateEmbedding,
} from './embed.js';

export type {
  EmbedConfig,
  ValidationResult,
} from './embed.js';

// ============================================================================
// Weight Buffer Management
// ============================================================================

export {
  isLayerWeights,
  getLayerWeights,
  getWeightBuffer,
  getNormWeightBuffer,
  getGPUWeightBuffer,
  createWeightBufferHelpers,
  BatchBufferTracker,
} from './weights.js';

export type {
  WeightBufferConfig,
  WeightDebugFlags,
} from './weights.js';

// ============================================================================
// Attention
// ============================================================================

export {
  runLayerAttentionGPU,
} from './attention.js';

export type {
  AttentionConfig,
  AttentionState,
  AttentionDebugFlags,
} from './attention.js';

// ============================================================================
// Feed-Forward Network
// ============================================================================

export {
  runFFNGPU,
  runFFN,
  layerNormCPU,
  gatherTokensCPU,
} from './ffn.js';

export type {
  FFNConfig,
  FFNWeights,
} from './ffn.js';

// ============================================================================
// Mixture of Experts
// ============================================================================

export {
  moeFeedForwardCPU,
  moeFeedForwardGPU,
  isMoELayer as isMoELayerFromImpl,
} from './moe-impl.js';

export type {
  MoEConfig,
  MoEExpertWeights,
  LayerRouterWeights,
  ExpertLoader,
} from './moe-impl.js';

// ============================================================================
// Layer Processing
// ============================================================================

export {
  processLayer,
  processLayerGPU,
  processLayerCPU,
  detectSandwichNorm,
  isMoELayer,
} from './layer.js';

export type {
  LayerContext,
  LayerResult,
  SandwichNormInfo,
} from './layer.js';

// ============================================================================
// Logits Computation
// ============================================================================

export {
  computeLogits,
  computeLogitsGPU,
  rmsNormCPU,
  matmulCPU,
  extractLastPositionLogits,
} from './logits.js';

export type {
  LogitsConfig,
  LogitsWeights,
  LogitsDebugFlags,
} from './logits.js';

// ============================================================================
// Initialization
// ============================================================================

export {
  normalizeAttentionKernel,
  initRoPEFrequencies,
  createKVCache,
  initTokenizer,
  loadWeights,
  applyGemmaChatTemplate,
  applyLlama3ChatTemplate,
  isStopToken as isStopTokenFromInit,
  initMoERouter,
  initSpeculativeDecoder,
} from './init.js';

// Note: init.ts types are internal implementation details

