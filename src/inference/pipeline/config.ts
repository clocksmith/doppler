/**
 * Model configuration parsing and normalization.
 * Handles HuggingFace, GGUF, and llama.cpp config formats.
 *
 * This module now uses the preset-based config-as-code system:
 * - JSON presets in config/presets/models/*.json define model family defaults
 * - resolveConfig() merges preset defaults with manifest overrides
 * - toParsedConfig() adapts ResolvedConfigSchema to ParsedModelConfig
 *
 * For legacy code, parseModelConfig() still works but now uses presets internally.
 *
 * See: config/loader.ts, config/presets/models/, config/schema/
 */

import { log } from '../../debug/index.js';
import type { LayerPipelineSchema, ResolvedConfigSchema } from '../../config/schema/index.js';
import { resolveConfig } from '../../config/loader.js';

export type ActivationType = 'silu' | 'gelu';

export interface RawConfig {
  model_type?: string;
  text_config?: RawConfig;
  architectures?: string[];
  hidden_size?: number;
  n_embd?: number;
  embeddingLength?: number;
  num_hidden_layers?: number;
  n_layer?: number;
  blockCount?: number;
  num_attention_heads?: number;
  n_head?: number;
  attentionHeadCount?: number;
  num_key_value_heads?: number;
  attentionHeadCountKV?: number;
  head_dim?: number;
  intermediate_size?: number;
  n_inner?: number;
  feedForwardLength?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  contextLength?: number;
  rope_theta?: number;
  rope_local_base_freq?: number;  // Gemma 3: Different RoPE theta for local/sliding attention layers
  ropeFreqBase?: number;
  rms_norm_eps?: number;
  attentionLayerNormRMSEpsilon?: number;
  hidden_activation?: string;
  hidden_act?: string;
  eos_token_id?: number | number[];
  rope_scaling?: RopeScalingConfig;
  sliding_window?: number;
  sliding_window_pattern?: number;  // Gemma 3: ratio of local:global layers (e.g., 6 = every 6th is global)
  num_local_experts?: number;
  num_experts?: number;
  experts_per_token?: number;
  num_experts_per_tok?: number;
  top_k?: number;
  layer_types?: string[];
  attention_bias?: boolean;
  quantization_config?: { quant_method?: string };
  // DOPPLER-specific config (can be set in manifest to override family defaults)
  scale_embeddings?: boolean;        // Gemma: scale by sqrt(hiddenSize)
  rms_norm_weight_offset?: boolean;  // Gemma 2/3: add +1 to norm weights
  // Gemma 2 softcapping (tanh-based value capping)
  final_logit_softcapping?: number;  // Gemma 2: 30.0, Gemma 3: null
  attn_logit_softcapping?: number;   // Gemma 2: 50.0, Gemma 3: null
  // Gemma 2 attention scaling (NOT sqrt(head_dim), uses head_dim directly)
  query_pre_attn_scalar?: number;    // Gemma 2: 256 (head_dim), standard: sqrt(head_dim)
}

export interface RopeScalingConfig {
  type?: string;
  rope_type?: string;
  factor?: number;
  beta_fast?: number;
  beta_slow?: number;
  original_max_position_embeddings?: number;
}

export interface TensorInfo {
  shape?: number[];
  dtype?: string;
}

export interface Manifest {
  architecture?: string;
  config?: RawConfig | Record<string, unknown>;
  tensors?: Record<string, TensorInfo>;
  tokenizer?: Record<string, unknown> & { vocab_size?: number };
  quantization?: string;
  eos_token_id?: number | number[];
  modelId?: string;
  model_id?: string;
  name?: string;
  draftModel?: { numTokens?: number };
  // RDRR manifest extensions
  optimizations?: {
    useBatching?: boolean;
    attentionKernel?: string;
    debug?: boolean;
    kernelHints?: Record<string, unknown>;
  };
  runtime?: {
    useBatching?: boolean;
    attentionKernel?: string;
    debug?: boolean;
  };
  attentionKernel?: string;
  // Quantization info with runtime hints
  quantizationInfo?: {
    weights?: string;
    embeddings?: string;
    lmHead?: string;
    compute?: string;  // Runtime compute precision hint (f16, f32)
  };
}

export interface AttentionParams {
  numHeads: number;
  numKVHeads: number;
  headDim: number;
}

export interface ParsedModelConfig {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  vocabSize: number;
  maxSeqLen: number;
  useMoE: boolean;
  numExperts: number;
  moeTopK: number;
  slidingWindow: number | null;
  ropeTheta: number;
  ropeLocalTheta: number | null;  // For local/sliding attention layers (Gemma 3: 10K vs 1M global)
  ropeScale: number;
  ropeScalingType: string | null;
  ropeScaling: RopeScalingConfig | null;
  quantization: string;
  quantMethod: string | null;
  rmsNormEps: number;
  rmsNormWeightOffset: boolean;
  scaleEmbeddings: boolean;
  hiddenActivation: ActivationType;
  isGemma3: boolean;
  isGemma2: boolean;
  isLlama3Instruct: boolean;
  isQwen3: boolean;
  isGptOss: boolean;
  stopTokenIds: number[];
  layerTypes: string[] | null;
  attentionBias: boolean;
  embeddingScale?: number;
  // Gemma 2 softcapping
  finalLogitSoftcapping: number | null;  // Gemma 2: 30.0
  attnLogitSoftcapping: number | null;   // Gemma 2: 50.0
  // Gemma 2 attention scaling: uses head_dim (256) instead of sqrt(head_dim) (16)
  queryPreAttnScalar: number;            // Gemma 2: 256, standard: sqrt(head_dim)
  // Optional layer pipeline override from model presets
  layerPipeline?: LayerPipelineSchema | null;
  // Chat template type from preset (gemma, llama3, gpt-oss, or null)
  chatTemplateType?: string | null;
}

// =============================================================================
// Model Detection Functions (kept for backward compatibility and edge cases)
// =============================================================================

export function isGemma3Model(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? config?.architectures?.[0] ?? '';
  const modelType = config?.model_type ?? config?.text_config?.model_type ?? '';
  return /gemma.*3|gemma3/i.test(arch) || /gemma.*3|gemma3/i.test(modelType);
}

export function isGemma2Model(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? config?.architectures?.[0] ?? '';
  const modelType = config?.model_type ?? config?.text_config?.model_type ?? '';
  return /gemma.*2|gemma2/i.test(arch) || /gemma.*2|gemma2/i.test(modelType);
}

export function isLlama3InstructModel(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? config?.architectures?.[0] ?? '';
  const modelId = manifest?.modelId ?? '';
  const isLlama3 = /llama.*3|llama3/i.test(arch) || /llama.*3|llama3/i.test(modelId);
  const isInstruct = /instruct|chat|it(?:-|$)/i.test(modelId);
  return isLlama3 && isInstruct;
}

export function isQwen3Model(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? config?.architectures?.[0] ?? '';
  const modelType = config?.model_type ?? config?.text_config?.model_type ?? '';
  return /qwen.*3|qwen3/i.test(arch) || /qwen.*3|qwen3/i.test(modelType);
}

export function isKimiK2Model(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? config?.architectures?.[0] ?? '';
  const modelType = config?.model_type ?? config?.text_config?.model_type ?? '';
  return /kimi.*k2|kimi_k2/i.test(arch) || /kimi.*k2|kimi_k2/i.test(modelType);
}

export function isMixtralModel(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? config?.architectures?.[0] ?? '';
  const modelType = config?.model_type ?? config?.text_config?.model_type ?? '';
  return /mixtral/i.test(arch) || /mixtral/i.test(modelType);
}

export function isGptOssModel(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? '';
  const modelType = config?.model_type ?? '';
  return /gpt.*oss|gptoss/i.test(arch) || /gpt.*oss|gptoss/i.test(modelType);
}

export function normalizeActivation(activation: string | undefined): ActivationType {
  if (!activation) return 'silu';
  const lower = activation.toLowerCase();
  if (lower.includes('gelu')) return 'gelu';
  if (lower.includes('silu') || lower.includes('swish')) return 'silu';
  return 'silu';
}

export function getStopTokenIds(config: RawConfig, manifest: Manifest): number[] {
  const eosTokenId = manifest?.eos_token_id ?? config?.eos_token_id ?? config?.text_config?.eos_token_id;
  if (Array.isArray(eosTokenId)) return eosTokenId;
  if (typeof eosTokenId === 'number') return [eosTokenId];
  if (isGemma3Model(config, manifest)) return [1, 106];
  return [];
}

// =============================================================================
// Tensor Inference Functions
// =============================================================================

export function inferAttentionParams(
  manifest: Manifest,
  hiddenSize: number,
  knownNumHeads: number | null = null
): AttentionParams | null {
  const tensors = manifest?.tensors ?? {};

  let qShape: number[] | undefined;
  let kShape: number[] | undefined;

  for (const [name, tensor] of Object.entries(tensors)) {
    const lower = name.toLowerCase();
    if (lower.includes('q_proj') || lower.includes('self_attn.q') || lower.includes('attn_q.weight')) {
      qShape = tensor?.shape;
    }
    if (lower.includes('k_proj') || lower.includes('self_attn.k') || lower.includes('attn_k.weight')) {
      kShape = tensor?.shape;
    }
    if (qShape && kShape) break;
  }

  if (!qShape || !kShape) return null;

  const qOutDim = qShape[0] === hiddenSize ? qShape[1] : qShape[0];
  const kOutDim = kShape[0] === hiddenSize ? kShape[1] : kShape[0];

  if (knownNumHeads && qOutDim % knownNumHeads === 0) {
    const headDim = qOutDim / knownNumHeads;
    if (kOutDim % headDim === 0) {
      const numKVHeads = kOutDim / headDim;
      if (numKVHeads > 0 && knownNumHeads >= numKVHeads) {
        return { numHeads: knownNumHeads, numKVHeads, headDim };
      }
    }
  }

  // Try q_norm weight for headDim
  for (const [name, tensor] of Object.entries(tensors)) {
    if ((name.includes('q_norm') || name.includes('attn_q_norm')) && tensor?.shape?.length === 1) {
      const normHeadDim = tensor.shape[0];
      if (qOutDim % normHeadDim === 0 && kOutDim % normHeadDim === 0) {
        const numHeads = qOutDim / normHeadDim;
        const numKVHeads = kOutDim / normHeadDim;
        if (numHeads >= numKVHeads && numHeads > 0 && numKVHeads > 0) {
          return { numHeads, numKVHeads, headDim: normHeadDim };
        }
      }
    }
  }

  // Try common headDim values
  for (const testHeadDim of [256, 128, 64, 96, 80, 160]) {
    if (qOutDim % testHeadDim === 0 && kOutDim % testHeadDim === 0) {
      const numHeads = qOutDim / testHeadDim;
      const numKVHeads = kOutDim / testHeadDim;
      if (numHeads >= numKVHeads && numHeads > 0 && numKVHeads > 0) {
        return { numHeads, numKVHeads, headDim: testHeadDim };
      }
    }
  }

  // Fallback
  const fallbackHeadDim = Math.floor(hiddenSize / 32);
  if (qOutDim % fallbackHeadDim === 0 && kOutDim % fallbackHeadDim === 0) {
    return {
      numHeads: qOutDim / fallbackHeadDim,
      numKVHeads: kOutDim / fallbackHeadDim,
      headDim: fallbackHeadDim,
    };
  }

  return null;
}

export function inferVocabSize(manifest: Manifest): number | null {
  const tensors = manifest?.tensors ?? {};

  for (const [name, tensor] of Object.entries(tensors)) {
    const lower = name.toLowerCase();
    const isEmbedding =
      lower.includes('embed_tokens.weight') ||
      lower.endsWith('wte.weight') ||
      lower.endsWith('tok_embeddings.weight') ||
      lower.endsWith('word_embeddings.weight') ||
      lower.endsWith('token_embd.weight');
    const isLmHead = lower.includes('lm_head.weight') || lower.endsWith('output.weight');

    if (!isEmbedding && !isLmHead) continue;

    const shape = tensor?.shape;
    if (!Array.isArray(shape) || shape.length === 0) continue;

    const vocabSize = Math.max(...shape);
    if (vocabSize > 1000) return vocabSize;
  }

  return null;
}

// =============================================================================
// Preset-Based Config Adapter
// =============================================================================

/**
 * Convert ResolvedConfigSchema to ParsedModelConfig.
 *
 * This adapter function enables gradual migration from the legacy if-statement
 * based parseModelConfig() to the preset-based resolveConfig() system.
 *
 * @param resolved - ResolvedConfigSchema from resolveConfig()
 * @param manifest - Original manifest (for fields not yet in presets)
 * @returns ParsedModelConfig compatible with existing pipeline code
 */
export function toParsedConfig(
  resolved: ResolvedConfigSchema,
  manifest: Manifest
): ParsedModelConfig {
  const rawConfig = (manifest.config ?? {}) as RawConfig;
  const config: RawConfig = rawConfig.text_config ?? rawConfig;
  const arch = resolved.architecture;
  const inf = resolved.inference;

  // Compute layer types from layerPattern
  let layerTypes: string[] | null = null;
  if (inf.layerPattern?.type === 'alternating') {
    const numLayers = arch.numLayers;
    const pattern = inf.layerPattern.globalPattern;
    const patternN = inf.layerPattern.globalPatternN ?? 6;

    if (pattern === 'even') {
      layerTypes = Array.from({ length: numLayers }, (_, i) =>
        i % 2 === 0 ? 'full_attention' : 'sliding_attention'
      );
    } else if (pattern === 'odd') {
      layerTypes = Array.from({ length: numLayers }, (_, i) =>
        i % 2 === 1 ? 'full_attention' : 'sliding_attention'
      );
    } else if (pattern === 'every_n') {
      layerTypes = Array.from({ length: numLayers }, (_, i) =>
        i % patternN === 0 ? 'full_attention' : 'sliding_attention'
      );
    }
  }

  // Compute queryPreAttnScalar
  const headDim = arch.headDim;
  const isGemma2 = resolved.preset === 'gemma2';
  const queryPreAttnScalar = config.query_pre_attn_scalar ?? (isGemma2 ? headDim : Math.sqrt(headDim));

  // Get stop token IDs from manifest/config (not yet in presets)
  const stopTokenIds = getStopTokenIds(config, manifest);

  // Get MoE config from manifest (not yet in presets)
  const useMoE = (config.num_local_experts ?? 0) > 1 || (config.num_experts ?? 0) > 1;
  const numExperts = config.num_local_experts ?? config.num_experts ?? 8;
  const moeTopK = config.experts_per_token ?? config.num_experts_per_tok ?? config.top_k ?? 2;

  // RoPE scaling config (combine preset + manifest)
  const ropeScaling = config.rope_scaling;
  let ropeScale = inf.rope?.ropeScalingFactor ?? 1.0;
  let ropeScalingType: string | null = inf.rope?.ropeScalingType ?? null;
  if (ropeScaling && typeof ropeScaling === 'object') {
    ropeScalingType = ropeScalingType ?? ropeScaling.type ?? ropeScaling.rope_type ?? null;
    const factor = ropeScaling.factor;
    if (factor && factor > 0) ropeScale = factor;
  }

  // Determine if this is a Gemma family model for scaleEmbeddings
  const isGemmaFamily = resolved.preset === 'gemma2' || resolved.preset === 'gemma3' || resolved.preset === 'functiongemma';

  const resolvedActivation =
    (inf.ffn?.activation as ActivationType | undefined) ??
    normalizeActivation(config.hidden_activation ?? config.hidden_act);

  return {
    numLayers: arch.numLayers,
    hiddenSize: arch.hiddenSize,
    intermediateSize: arch.intermediateSize,
    numHeads: arch.numAttentionHeads,
    numKVHeads: arch.numKeyValueHeads,
    headDim: arch.headDim,
    vocabSize: arch.vocabSize,
    maxSeqLen: arch.maxSeqLen,
    useMoE,
    numExperts,
    moeTopK,
    slidingWindow: inf.attention?.slidingWindow ?? null,
    ropeTheta: inf.rope?.ropeTheta ?? arch.ropeTheta ?? 10000,
    ropeLocalTheta: inf.rope?.ropeLocalTheta ?? null,
    ropeScale,
    ropeScalingType,
    ropeScaling: ropeScaling ? { ...ropeScaling, factor: ropeScale } : null,
    quantization: (manifest.quantization as string) ?? 'f16',
    quantMethod: config.quantization_config?.quant_method ?? null,
    rmsNormEps: inf.normalization?.rmsNormEps ?? arch.rmsNormEps ?? 1e-5,
    rmsNormWeightOffset: inf.normalization?.rmsNormWeightOffset ?? false,
    scaleEmbeddings: config.scale_embeddings ?? isGemmaFamily,
    hiddenActivation: resolvedActivation,
    isGemma3: resolved.preset === 'gemma3',
    isGemma2: resolved.preset === 'gemma2',
    isLlama3Instruct: resolved.preset === 'llama3',
    isQwen3: resolved.preset === 'qwen3',
    isGptOss: false,
    stopTokenIds,
    layerTypes,
    attentionBias: config.attention_bias ?? false,
    finalLogitSoftcapping: inf.output?.finalLogitSoftcapping ?? null,
    attnLogitSoftcapping: inf.attention?.attnLogitSoftcapping ?? null,
    queryPreAttnScalar,
    layerPipeline: inf.pipeline ?? null,
    chatTemplateType: inf.chatTemplate?.type ?? null,
  };
}

/**
 * Convert pipeline Manifest to config ManifestSchema for resolveConfig().
 */
function toManifestSchema(manifest: Manifest): import('../../config/schema/index.js').ManifestSchema {
  return {
    version: 1,
    modelId: manifest.modelId ?? manifest.model_id ?? 'unknown',
    modelType: (manifest.architecture as string) ?? 'transformer',
    quantization: manifest.quantization ?? 'f16',
    hashAlgorithm: 'sha256' as const,
    totalSize: 0,
    architecture: manifest.architecture ?? 'transformer',
    shards: [],
    config: manifest.config as Record<string, unknown>,
    tokenizer: manifest.tokenizer as unknown as import('../../config/schema/index.js').TokenizerSchema | undefined,
  };
}

// =============================================================================
// Main Entry Point - Uses Preset-Based Resolution
// =============================================================================

/**
 * Parse model configuration from manifest using preset-based resolution.
 *
 * This is the main entry point that uses the config-as-code preset system.
 * It internally calls resolveConfig() and then converts to ParsedModelConfig.
 *
 * @param manifest - Model manifest
 * @returns ParsedModelConfig for pipeline consumption
 */
export function parseModelConfig(manifest: Manifest): ParsedModelConfig {
  // Convert pipeline Manifest to config ManifestSchema
  const manifestSchema = toManifestSchema(manifest);

  // Use preset-based resolution
  const resolved = resolveConfig(manifestSchema);

  // Log which preset was detected
  log.debug('Config', `Detected preset: ${resolved.preset}`);

  // Convert to ParsedModelConfig for backward compatibility
  const parsed = toParsedConfig(resolved, manifest);

  // Attention params inference still needed for some models
  const rawConfig = (manifest.config ?? {}) as RawConfig;
  const config: RawConfig = rawConfig.text_config ?? rawConfig;

  // Check if we need to infer attention params
  let numHeads = parsed.numHeads;
  let numKVHeads = parsed.numKVHeads;
  let headDim = parsed.headDim;

  const hasConfiguredHeads = config.num_attention_heads ?? config.n_head ?? config.attentionHeadCount;
  if (!hasConfiguredHeads && manifest.tensors) {
    log.debug('Config', `parseModelConfig: inferring attention params from tensors`);
    const inferred = inferAttentionParams(manifest, parsed.hiddenSize, null);
    if (inferred) {
      numHeads = inferred.numHeads;
      numKVHeads = inferred.numKVHeads;
      headDim = inferred.headDim;
      log.debug('Config', `Inferred attention params: numHeads=${numHeads}, numKVHeads=${numKVHeads}, headDim=${headDim}`);
    }
  }

  // Infer vocab size if needed
  let vocabSize = parsed.vocabSize;
  const vocabCandidates: number[] = [];
  if (config.vocab_size && config.vocab_size > 0) vocabCandidates.push(config.vocab_size);
  if (manifest.tokenizer?.vocab_size) vocabCandidates.push(manifest.tokenizer.vocab_size);
  const inferredVocab = inferVocabSize(manifest);
  if (inferredVocab) vocabCandidates.push(inferredVocab);
  if (vocabCandidates.length > 0) {
    vocabSize = Math.max(...vocabCandidates);
  }

  // Handle Gemma 2 special case for head_dim
  const isGemma2Early = isGemma2Model(rawConfig, manifest);
  if (!config.head_dim && isGemma2Early && parsed.hiddenSize === 3584 && numHeads === 16) {
    headDim = 256;
    log.debug('Config', 'Gemma 2 9B detected: using head_dim=256 (expanded attention architecture)');
  }

  // Build final config with inferred values
  const finalConfig: ParsedModelConfig = {
    ...parsed,
    numHeads,
    numKVHeads,
    headDim,
    vocabSize,
    queryPreAttnScalar: config.query_pre_attn_scalar ?? (isGemma2Early ? headDim : Math.sqrt(headDim)),
  };

  // Log critical Gemma 2 settings
  if (finalConfig.isGemma2) {
    log.info('Config', `Gemma 2: ropeTheta=${finalConfig.ropeTheta}, ropeLocalTheta=${finalConfig.ropeLocalTheta}, slidingWindow=${finalConfig.slidingWindow}, attnSoftcap=${finalConfig.attnLogitSoftcapping}, logitSoftcap=${finalConfig.finalLogitSoftcapping}, queryPreAttnScalar=${finalConfig.queryPreAttnScalar}`);
  }

  return finalConfig;
}
