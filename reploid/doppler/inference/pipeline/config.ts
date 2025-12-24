/**
 * Model configuration parsing and normalization.
 * Handles HuggingFace, GGUF, and llama.cpp config formats.
 */

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
  rms_norm_weight_offset?: boolean;  // Gemma 3: add +1 to norm weights
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
  isLlama3Instruct: boolean;
  isQwen3: boolean;
  isGptOss: boolean;
  stopTokenIds: number[];
  layerTypes: string[] | null;
  attentionBias: boolean;
  embeddingScale?: number;
}

export function isGemma3Model(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? config?.architectures?.[0] ?? '';
  const modelType = config?.model_type ?? config?.text_config?.model_type ?? '';
  return /gemma/i.test(arch) || /gemma/i.test(modelType);
}

export function isLlama3InstructModel(config: RawConfig, manifest: Manifest): boolean {
  const arch = manifest?.architecture ?? config?.architectures?.[0] ?? '';
  const modelId = manifest?.modelId ?? '';
  // Check for Llama 3 architecture and instruction-tuned variant
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
    // Match HuggingFace (q_proj, self_attn.q) and GGUF (attn_q) naming patterns
    if (lower.includes('q_proj') || lower.includes('self_attn.q') || lower.includes('attn_q.weight')) qShape = tensor?.shape;
    if (lower.includes('k_proj') || lower.includes('self_attn.k') || lower.includes('attn_k.weight')) kShape = tensor?.shape;
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
      lower.endsWith('token_embd.weight');  // GGUF naming
    const isLmHead = lower.includes('lm_head.weight') || lower.endsWith('output.weight');

    if (!isEmbedding && !isLmHead) continue;

    const shape = tensor?.shape;
    if (!Array.isArray(shape) || shape.length === 0) continue;

    const vocabSize = Math.max(...shape);
    if (vocabSize > 1000) return vocabSize;
  }

  return null;
}

export function parseModelConfig(manifest: Manifest): ParsedModelConfig {
  const rawConfig = (manifest.config ?? {}) as RawConfig;
  const config: RawConfig = rawConfig.text_config ?? rawConfig;

  // Normalize GGUF camelCase to snake_case
  const hiddenSize = config.hidden_size ?? config.n_embd ?? config.embeddingLength ?? 4096;
  const numLayers = config.num_hidden_layers ?? config.n_layer ?? config.blockCount ?? 32;
  const intermediateSize = config.intermediate_size ?? config.n_inner ?? config.feedForwardLength ?? hiddenSize * 4;

  let numHeads = config.num_attention_heads ?? config.n_head ?? config.attentionHeadCount;
  let numKVHeads = config.num_key_value_heads ?? config.attentionHeadCountKV;
  let headDim = config.head_dim;

  // Vocab size from multiple sources
  const vocabCandidates: number[] = [];
  if (config.vocab_size && config.vocab_size > 0) vocabCandidates.push(config.vocab_size);
  if (manifest.tokenizer?.vocab_size) vocabCandidates.push(manifest.tokenizer.vocab_size);
  const inferredVocab = inferVocabSize(manifest);
  if (inferredVocab) vocabCandidates.push(inferredVocab);
  const vocabSize = vocabCandidates.length > 0 ? Math.max(...vocabCandidates) : 32000;

  // Infer attention params if missing
  if (!numHeads || !headDim) {
    const inferred = inferAttentionParams(manifest, hiddenSize, numHeads ?? null);
    if (inferred) {
      numHeads = numHeads ?? inferred.numHeads;
      numKVHeads = numKVHeads ?? inferred.numKVHeads;
      headDim = headDim ?? inferred.headDim;
    }
  }

  numHeads = numHeads ?? 32;
  numKVHeads = numKVHeads ?? numHeads;
  headDim = headDim ?? Math.floor(hiddenSize / numHeads);

  // RoPE scaling
  const ropeScaling = config.rope_scaling;
  let ropeScale = 1.0;
  let ropeScalingType: string | null = null;
  if (ropeScaling && typeof ropeScaling === 'object') {
    ropeScalingType = ropeScaling.type ?? ropeScaling.rope_type ?? 'linear';
    const factor = ropeScaling.factor;
    if (factor && factor > 0) ropeScale = factor;
  }

  // Model family detection (used for defaults, not hard branches)
  const isGemma3 = isGemma3Model(rawConfig, manifest);
  const isLlama3Instruct = isLlama3InstructModel(rawConfig, manifest);
  const isQwen3 = isQwen3Model(rawConfig, manifest);
  const isKimiK2 = isKimiK2Model(rawConfig, manifest);
  const isMixtral = isMixtralModel(rawConfig, manifest);
  const isGptOss = isGptOssModel(rawConfig, manifest);

  // Config values with family-based defaults
  // Explicit config > family default > universal default
  const rmsNormEps = config.rms_norm_eps ?? config.attentionLayerNormRMSEpsilon ?? (isGemma3 ? 1e-6 : 1e-5);
  // Gemma 3 uses GELU activation, not SiLU (affects FFN gate activation)
  const hiddenActivation = normalizeActivation(config.hidden_activation ?? config.hidden_act ?? (isGemma3 ? 'gelu' : undefined));
  const moeTopK = config.experts_per_token ?? config.num_experts_per_tok ?? config.top_k ?? 2;

  // DOPPLER-specific architectural quirks (explicit config overrides family detection)
  const scaleEmbeddings = config.scale_embeddings ?? isGemma3;
  const rmsNormWeightOffset = config.rms_norm_weight_offset ?? isGemma3;

  // RoPE theta defaults by architecture:
  // - Gemma 3, Qwen3, Mixtral: 1,000,000
  // - Kimi K2: 50,000 (with YARN scaling)
  // - Others: 10,000
  let defaultRopeTheta = 10000;
  if (isGemma3 || isQwen3 || isMixtral) {
    defaultRopeTheta = 1000000;
  } else if (isKimiK2) {
    defaultRopeTheta = 50000;  // Uses YARN 32x scaling for 128K context
  }

  // Generate layer_types for Gemma 3 if not present in config
  // Gemma 3 uses sliding_window_pattern to determine which layers are global (full_attention)
  // Every Nth layer is global (full_attention), others are local (sliding_attention)
  // Pattern: layer 0, N, 2N, 3N... are global; others are local
  let layerTypes: string[] | null = Array.isArray(config.layer_types) ? config.layer_types : null;
  if (!layerTypes && isGemma3) {
    const pattern = config.sliding_window_pattern ?? 6;  // Default to 6 if not specified
    layerTypes = Array.from({ length: numLayers }, (_, i) =>
      i % pattern === 0 ? 'full_attention' : 'sliding_attention'
    );
  }

  // RoPE local theta for sliding attention layers (Gemma 3: 10K vs 1M global)
  const ropeLocalTheta = config.rope_local_base_freq ?? null;

  return {
    numLayers,
    hiddenSize,
    intermediateSize,
    numHeads,
    numKVHeads,
    headDim,
    vocabSize,
    maxSeqLen: config.max_position_embeddings ?? config.contextLength ?? 4096,
    useMoE: (config.num_local_experts ?? 0) > 1 || (config.num_experts ?? 0) > 1,
    numExperts: config.num_local_experts ?? config.num_experts ?? 8,
    moeTopK,
    slidingWindow: config.sliding_window ?? null,
    ropeTheta: config.rope_theta ?? config.ropeFreqBase ?? defaultRopeTheta,
    ropeLocalTheta,
    ropeScale,
    ropeScalingType,
    ropeScaling: ropeScaling ? { ...ropeScaling, factor: ropeScale } : null,
    quantization: (manifest.quantization as string) ?? 'f16',
    quantMethod: config.quantization_config?.quant_method ?? null,
    rmsNormEps,
    rmsNormWeightOffset,
    scaleEmbeddings,
    hiddenActivation,
    isGemma3,
    isLlama3Instruct,
    isQwen3,
    isGptOss,
    stopTokenIds: getStopTokenIds(rawConfig, manifest),
    layerTypes,
    attentionBias: config.attention_bias ?? false,
  };
}
