import {
  // Constants
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  RDRR_VERSION as SCHEMA_RDRR_VERSION,
  ConversionStage as SchemaConversionStage,
  DEFAULT_MANIFEST_INFERENCE,
  formatBytes,
} from '../config/schema/index.js';
import {
  classifyTensorRole,
  generateShardFilename,
  resolveTensorGroup,
  resolveTensorRole,
} from '../formats/rdrr/index.js';
import { log } from '../debug/index.js';
import {
  getInferenceLayerPatternContractArtifact,
  selectRuleValue,
} from '../rules/rule-registry.js';
import {
  createConverterConfig,
} from '../config/index.js';
import { buildExecutionContractArtifact } from '../config/execution-contract-check.js';
import { buildManifestRequiredInferenceFieldsArtifact } from '../config/required-inference-fields-contract-check.js';
import { resolveEosTokenId } from './tokenizer-special-tokens.js';
import { inferBundledTokenizerBehaviorFlags } from '../inference/tokenizers/behavior-flags.js';
import {
  normalizeQ4KLayout,
  resolveManifestQuantization,
  resolveEffectiveQuantizationInfo,
} from './quantization-info.js';
import {
  float16ToFloat32,
  float32ToFloat16,
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  quantizeToQ4KMColumnWise,
  quantizeToInt4PerRowSymmetric,
} from './quantizer.js';
import { cloneJsonValue } from '../formats/clone-json.js';
import { SOURCE_PACKED_QUANT_DTYPES, isGemma4PerLayerEmbedTensor, normalizeModulesToNotConvert, normalizeStorageQuant, resolveQuantizeEmbeddings, transformTensorBytes } from './tensor-transform.js';
import { buildArtifactIdentity, extractArchitecture, getNestedTextConfig, sanitizeModelId } from './artifact-identity.js';

export const RDRR_VERSION = SCHEMA_RDRR_VERSION;

export const EMBEDDING_TENSOR_NAMES = [
  'language_model.model.embed_tokens.weight',
  'model.embed_tokens.weight',
  'embed_tokens.weight',
  'token_embd.weight',
  'wte.weight',
  'transformer.wte.weight',
];

export function inferEmbeddingOutputConfig(tensorLocations) {
  // Normalize Map input to a plain object so the rest of the function
  // handles a single type consistently.
  const normalized = tensorLocations instanceof Map
    ? Object.fromEntries(tensorLocations)
    : tensorLocations ?? {};

  const getLocation = (name) => normalized[name];

  const entries = Object.entries(normalized);
  for (const [_name, loc] of entries) {
    if (loc?.role === 'embedding' && loc.shape?.length === 2) {
      const [dim0, dim1] = loc.shape;
      const isGGUFLayout = dim0 < dim1;
      return {
        embeddingTranspose: isGGUFLayout,
        embeddingVocabSize: isGGUFLayout ? dim1 : dim0,
      };
    }
  }

  for (const name of EMBEDDING_TENSOR_NAMES) {
    const loc = getLocation(name);
    if (loc?.shape && loc.shape.length === 2) {
      const [dim0, dim1] = loc.shape;
      const isGGUFLayout = dim0 < dim1;
      return {
        embeddingTranspose: isGGUFLayout,
        embeddingVocabSize: isGGUFLayout ? dim1 : dim0,
      };
    }
  }

  return null;
}

export function resolveTokenizerId(value) {
  if (typeof value === 'number') return value;
  return null;
}

export function resolveTokenizerIds(value) {
  if (Array.isArray(value) && value.every((id) => typeof id === 'number')) {
    return value;
  }
  if (typeof value === 'number') return [value];
  return null;
}

export function resolveTokenizerField(tokenizerConfig, ...keys) {
  if (!tokenizerConfig) return null;
  for (const key of keys) {
    if (tokenizerConfig[key] != null) {
      return tokenizerConfig[key];
    }
  }
  return null;
}

export function resolveConfigBoolean(rawConfig, ...keys) {
  // Same lookup logic as resolveTokenizerField: return the first non-nullish
  // value from the given keys. Delegates to avoid duplicating the pattern.
  return resolveTokenizerField(rawConfig, ...keys);
}

export function resolveTokenizerVocabSize(tokenizerConfig, rawConfig, architecture) {
  const nestedTextConfig = getNestedTextConfig(rawConfig);
  const configVocab = rawConfig?.vocab_size ?? nestedTextConfig?.vocab_size;
  const tokenizerVocab = tokenizerConfig?.vocab_size ?? tokenizerConfig?.vocabSize;
  const archVocab = architecture?.vocabSize;

  // Warn if multiple sources provide vocab size and they disagree
  const sources = [
    tokenizerVocab != null ? { label: 'tokenizer', value: tokenizerVocab } : null,
    configVocab != null ? { label: 'config', value: configVocab } : null,
    archVocab != null ? { label: 'architecture', value: archVocab } : null,
  ].filter(Boolean);
  if (sources.length > 1) {
    const distinct = new Set(sources.map((s) => s.value));
    if (distinct.size > 1) {
      const detail = sources.map((s) => `${s.label}=${s.value}`).join(', ');
      log.error(
        'Convert',
        `Vocab size sources disagree: ${detail}. Using first available (${sources[0].label}=${sources[0].value}). ` +
        'This may cause embedding size mismatches at runtime. Verify the correct vocab size in the conversion config.'
      );
    }
  }

  return tokenizerVocab ?? configVocab ?? archVocab ?? null;
}

export function resolveConfigTokenId(rawConfig, key) {
  const direct = rawConfig?.[key];
  const nested = getNestedTextConfig(rawConfig)?.[key];
  return resolveTokenizerId(direct ?? nested);
}

export function resolveConfigTokenIds(rawConfig, key) {
  const direct = rawConfig?.[key];
  const nested = getNestedTextConfig(rawConfig)?.[key];
  return resolveTokenizerIds(direct ?? nested);
}

export function resolveMoEConfigNumber(rawConfig, ...keys) {
  const nestedTextConfig = getNestedTextConfig(rawConfig);
  for (const key of keys) {
    const direct = rawConfig?.[key];
    if (Number.isFinite(direct) && direct > 0) return Number(direct);
    const nested = nestedTextConfig?.[key];
    if (Number.isFinite(nested) && nested > 0) return Number(nested);
  }
  return null;
}

export function normalizeTensorShape(value) {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const rows = Number(value[0]);
  const cols = Number(value[1]);
  if (!Number.isFinite(rows) || !Number.isFinite(cols)) return null;
  if (rows <= 0 || cols <= 0) return null;
  return [Math.trunc(rows), Math.trunc(cols)];
}

export function isExpertTensorName(name) {
  const lower = String(name || '').toLowerCase();
  return lower.includes('.experts.') || lower.includes('.expert.') || lower.includes('block_sparse_moe');
}

export function inferDenseIntermediateSizeFromTensorEntries(entries) {
  if (!Array.isArray(entries) || entries.length === 0) return null;
  const candidates = [];
  for (const entry of entries) {
    const name = String(entry?.name || '');
    if (!name || isExpertTensorName(name)) continue;
    const shape = normalizeTensorShape(entry?.shape);
    if (!shape) continue;
    const lower = name.toLowerCase();
    if (
      lower.endsWith('.feed_forward.w1.weight')
      || lower.endsWith('.feed_forward.w3.weight')
      || lower.endsWith('.ffn_gate.weight')
      || lower.endsWith('.ffn_up.weight')
      || lower.endsWith('.ffn.gate_proj.weight')
      || lower.endsWith('.ffn.up_proj.weight')
      || lower.endsWith('.mlp.gate_proj.weight')
      || lower.endsWith('.mlp.up_proj.weight')
    ) {
      candidates.push(shape[0]);
      continue;
    }
    if (
      lower.endsWith('.feed_forward.w2.weight')
      || lower.endsWith('.ffn_down.weight')
      || lower.endsWith('.ffn.down_proj.weight')
      || lower.endsWith('.mlp.down_proj.weight')
    ) {
      candidates.push(shape[1]);
      continue;
    }
    if (
      lower.endsWith('.feed_forward.w1_w3.weight')
      || lower.endsWith('.ffn_gate_up.weight')
      || lower.endsWith('.ffn.gate_up_proj.weight')
      || lower.endsWith('.mlp.gate_up_proj.weight')
    ) {
      if (shape[0] % 2 === 0) {
        candidates.push(Math.trunc(shape[0] / 2));
      }
    }
  }
  if (candidates.length === 0) return null;
  const counts = new Map();
  for (const value of candidates) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0] - b[0];
    })[0]?.[0] ?? null;
}

export function resolveIntermediateSizeFromTensors(architecture, model, tensorLocations, rawConfig, modelId) {
  if (!architecture || typeof architecture !== 'object') return architecture;
  const current = architecture.intermediateSize;
  if (typeof current !== 'number' || !Number.isFinite(current) || current <= 0) {
    return architecture;
  }
  const modelType = String(rawConfig?.model_type ?? getNestedTextConfig(rawConfig)?.model_type ?? '').toLowerCase();
  if (modelType !== 'lfm2' && modelType !== 'amplify') {
    return architecture;
  }
  const entries = Array.isArray(model?.tensors) && model.tensors.length > 0
    ? model.tensors
    : Object.entries(tensorLocations ?? {}).map(([name, location]) => ({ name, shape: location?.shape }));
  const inferred = inferDenseIntermediateSizeFromTensorEntries(entries);
  if (inferred == null || inferred === current) {
    return architecture;
  }
  log.warn(
    'Convert',
    `Adjusted architecture.intermediateSize for "${modelId}": ${current} -> ${inferred} (from FFN tensor shapes)`
  );
  return {
    ...architecture,
    intermediateSize: inferred,
  };
}

export function modelHasMoETensors(model) {
  if (!Array.isArray(model?.tensors)) return false;
  return model.tensors.some((tensor) => {
    const name = String(tensor?.name || '').toLowerCase();
    return (
      name.includes('.experts.') ||
      name.includes('.expert.') ||
      name.includes('block_sparse_moe')
    );
  });
}

export function resolveMoEExpertFormat(rawConfig, resolvedModelType, quantizationInfo, explicitFormat) {
  if (explicitFormat) return explicitFormat;
  const fromQuant = quantizationInfo?.expertsFormat;
  if (typeof fromQuant === 'string' && fromQuant.length > 0) {
    return fromQuant;
  }
  const modelType = String(
    resolvedModelType ??
    rawConfig?.model_type ??
    getNestedTextConfig(rawConfig)?.model_type ??
    ''
  ).toLowerCase();
  if (modelType.includes('gpt_oss') || modelType.includes('gpt-oss') || modelType.includes('gptoss')) {
    return 'gpt-oss';
  }
  if (modelType === 'diffusion_gemma' || modelType === 'diffusion_gemma_text') {
    return 'gemma4';
  }
  return 'mixtral';
}

export function normalizeMoEConfig(config, contextLabel) {
  if (!config) return null;
  const numExperts = Number(config.numExperts);
  const numExpertsPerToken = Number(config.numExpertsPerToken);
  const expertFormat = String(config.expertFormat || '').trim();
  const expertIntermediateSize = config.expertIntermediateSize == null
    ? null
    : Number(config.expertIntermediateSize);
  const allowedExpertFormat = expertFormat === 'gpt-oss' || expertFormat === 'mixtral' || expertFormat === 'gemma4';
  if (!Number.isFinite(numExperts) || numExperts <= 0) {
    throw new Error(`Invalid moeConfig.numExperts for ${contextLabel}`);
  }
  if (!Number.isFinite(numExpertsPerToken) || numExpertsPerToken <= 0) {
    throw new Error(`Invalid moeConfig.numExpertsPerToken for ${contextLabel}`);
  }
  if (numExpertsPerToken > numExperts) {
    throw new Error(`Invalid moeConfig for ${contextLabel}: numExpertsPerToken cannot exceed numExperts`);
  }
  if (!allowedExpertFormat) {
    throw new Error(`Invalid moeConfig.expertFormat for ${contextLabel}: "${expertFormat}"`);
  }
  if (
    expertIntermediateSize != null
    && (!Number.isFinite(expertIntermediateSize) || expertIntermediateSize <= 0)
  ) {
    throw new Error(`Invalid moeConfig.expertIntermediateSize for ${contextLabel}`);
  }
  if (expertFormat === 'gemma4' && expertIntermediateSize == null) {
    throw new Error(`Invalid moeConfig for ${contextLabel}: gemma4 experts require expertIntermediateSize`);
  }
  return {
    numExperts,
    numExpertsPerToken,
    expertFormat,
    ...(expertIntermediateSize == null ? {} : { expertIntermediateSize }),
  };
}

export function resolveManifestMoEConfig(model, options, rawConfig, resolvedModelType) {
  const explicit = normalizeMoEConfig(options?.moeConfig ?? null, options?.modelId ?? 'model');
  if (explicit) return explicit;

  const hasMoETensors = modelHasMoETensors(model);
  const numExperts = resolveMoEConfigNumber(rawConfig, 'num_local_experts', 'num_experts', 'expertCount');

  // If the checkpoint does not expose MoE tensors and config does not declare experts,
  // this is a dense model and should not emit moeConfig.
  if (!hasMoETensors && (!numExperts || numExperts <= 1)) {
    return null;
  }

  if (!numExperts || numExperts <= 0) {
    throw new Error(
      `MoE tensors detected for "${options?.modelId ?? 'model'}" but expert count is missing in config`
    );
  }

  const numExpertsPerToken = resolveMoEConfigNumber(
    rawConfig,
    'top_k_experts',
    'num_experts_per_tok',
    'num_experts_per_token',
    'experts_per_token',
    'expertUsedCount'
  );

  if (!numExpertsPerToken) {
    throw new Error(
      `MoE model "${options?.modelId ?? 'model'}" missing experts-per-token config ` +
      '(expected top_k_experts/num_experts_per_tok/num_experts_per_token/experts_per_token)'
    );
  }

  const expertFormat = resolveMoEExpertFormat(
    rawConfig,
    resolvedModelType,
    options?.quantizationInfo ?? null,
    null
  );
  const expertIntermediateSize = expertFormat === 'gemma4'
    ? resolveMoEConfigNumber(rawConfig, 'moe_intermediate_size', 'expert_intermediate_size')
    : null;

  return normalizeMoEConfig(
    {
      numExperts,
      numExpertsPerToken,
      expertFormat,
      ...(expertIntermediateSize == null ? {} : { expertIntermediateSize }),
    },
    options?.modelId ?? 'model'
  );
}

export function buildSentencepieceTokenizer(tokenizerConfig, rawConfig, architecture, modelTokenizerModel) {
  if (!modelTokenizerModel) return null;

  const vocabSize = resolveTokenizerVocabSize(tokenizerConfig, rawConfig, architecture);
  const sentencepieceModel = typeof modelTokenizerModel === 'string'
    ? modelTokenizerModel
    : modelTokenizerModel?.file ?? 'tokenizer.model';

  const bosTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'bos_token_id', 'bosTokenId')
    ?? resolveConfigTokenId(rawConfig, 'bos_token_id')
  );
  const eosTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'eos_token_id', 'eosTokenId')
    ?? resolveConfigTokenId(rawConfig, 'eos_token_id')
  );
  const eosTokens = resolveTokenizerIds(
    resolveTokenizerField(tokenizerConfig, 'eos_token_ids', 'eosTokens', 'eos_token_id')
    ?? resolveConfigTokenIds(rawConfig, 'eos_token_ids')
  );
  const padTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'pad_token_id', 'padTokenId')
    ?? resolveConfigTokenId(rawConfig, 'pad_token_id')
  );
  const unkTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'unk_token_id', 'unkTokenId')
    ?? resolveConfigTokenId(rawConfig, 'unk_token_id')
  );
  const addBosToken = resolveTokenizerField(tokenizerConfig, 'add_bos_token', 'addBosToken');
  const addEosToken = resolveTokenizerField(tokenizerConfig, 'add_eos_token', 'addEosToken');

  const tokenizer = {
    type: 'sentencepiece',
    sentencepieceModel,
    vocabSize: vocabSize ?? 0,
  };

  if (bosTokenId != null) tokenizer.bosTokenId = bosTokenId;
  if (eosTokenId != null) tokenizer.eosTokenId = eosTokenId;
  if (eosTokens) tokenizer.eosTokens = eosTokens;
  if (padTokenId != null) tokenizer.padTokenId = padTokenId;
  if (unkTokenId != null) tokenizer.unkTokenId = unkTokenId;
  if (addBosToken != null) tokenizer.addBosToken = addBosToken;
  if (addEosToken != null) tokenizer.addEosToken = addEosToken;

  return tokenizer;
}

export function resolveBundledTokenizerVocabSize(tokenizerJson) {
  const vocab = tokenizerJson?.model?.vocab;
  if (Array.isArray(vocab)) {
    return vocab.length;
  }
  if (vocab && typeof vocab === 'object') {
    return Object.keys(vocab).length;
  }
  return 0;
}

export function buildBundledTokenizer(tokenizerJson, tokenizerConfig, rawConfig) {
  const vocabSize = resolveBundledTokenizerVocabSize(tokenizerJson);
  if (!vocabSize) {
    throw new Error('Tokenizer vocab is missing or empty');
  }

  const tokenizer = {
    type: 'bundled',
    vocabSize,
    file: 'tokenizer.json',
  };
  const addBosToken = (
    resolveTokenizerField(tokenizerJson, 'add_bos_token', 'addBosToken')
    ?? resolveTokenizerField(tokenizerConfig, 'add_bos_token', 'addBosToken')
    ?? resolveConfigBoolean(rawConfig, 'add_bos_token', 'addBosToken')
  );
  const addEosToken = (
    resolveTokenizerField(tokenizerJson, 'add_eos_token', 'addEosToken')
    ?? resolveTokenizerField(tokenizerConfig, 'add_eos_token', 'addEosToken')
    ?? resolveConfigBoolean(rawConfig, 'add_eos_token', 'addEosToken')
  );
  const inferredFlags = inferBundledTokenizerBehaviorFlags(tokenizerJson);

  if (addBosToken != null) tokenizer.addBosToken = addBosToken;
  else if (inferredFlags.addBosToken != null) tokenizer.addBosToken = inferredFlags.addBosToken;
  if (addEosToken != null) tokenizer.addEosToken = addEosToken;
  else if (inferredFlags.addEosToken != null) tokenizer.addEosToken = inferredFlags.addEosToken;

  return tokenizer;
}

export function resolveDiffusionGemmaConfig(rawConfig, resolvedModelType = null) {
  const textConfig = getNestedTextConfig(rawConfig);
  const modelType = String(
    resolvedModelType
    ?? rawConfig?.model_type
    ?? textConfig?.model_type
    ?? ''
  ).trim().toLowerCase();
  const textModelType = String(textConfig?.model_type ?? '').trim().toLowerCase();
  if (
    modelType !== 'diffusion_gemma'
    && modelType !== 'diffusion_gemma_text'
    && textModelType !== 'diffusion_gemma'
    && textModelType !== 'diffusion_gemma_text'
  ) {
    return null;
  }
  return textConfig ?? rawConfig ?? {};
}

export function readDiffusionGemmaPositiveInteger(value, label, modelId) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`DiffusionGemma model "${modelId}" requires ${label} to be a positive integer.`);
  }
  return value;
}

export function readDiffusionGemmaNonNegativeNumber(value, label, modelId) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`DiffusionGemma model "${modelId}" requires ${label} to be a non-negative finite number.`);
  }
  return value;
}

export function readDiffusionGemmaEntropyBound(generationConfig, modelId) {
  if (generationConfig && Object.hasOwn(generationConfig, 'entropy_bound')) {
    return readDiffusionGemmaNonNegativeNumber(
      generationConfig.entropy_bound,
      'generation_config.entropy_bound',
      modelId
    );
  }
  return readDiffusionGemmaNonNegativeNumber(
    generationConfig?.sampler_config?.entropy_bound,
    'generation_config.sampler_config.entropy_bound',
    modelId
  );
}

export function readDiffusionGemmaNullableTokenId(value, label, modelId) {
  if (value == null) return null;
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`DiffusionGemma model "${modelId}" requires ${label} to be null or a non-negative integer.`);
  }
  return value;
}

export function readDiffusionGemmaTokenId(value, label, modelId) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`DiffusionGemma model "${modelId}" requires ${label} to be a non-negative integer.`);
  }
  return value;
}

export function readDiffusionGemmaEosTokenIds(rawConfig, generationConfig, modelId) {
  const eos = resolveTokenizerIds(
    generationConfig?.eos_token_id
    ?? generationConfig?.eos_token_ids
    ?? rawConfig?.eos_token_id
    ?? rawConfig?.eos_token_ids
  );
  if (!Array.isArray(eos) || eos.length === 0) {
    throw new Error(`DiffusionGemma model "${modelId}" requires generation_config.eos_token_id.`);
  }
  return eos;
}

export function resolveDiffusionGemmaInferenceContract(rawConfig, generationConfig, modelId) {
  const canvasLength = readDiffusionGemmaPositiveInteger(
    rawConfig?.canvas_length,
    'config.canvas_length',
    modelId
  );
  const maxDenoisingSteps = readDiffusionGemmaPositiveInteger(
    generationConfig?.max_denoising_steps,
    'generation_config.max_denoising_steps',
    modelId
  );
  const maxNewTokens = readDiffusionGemmaPositiveInteger(
    generationConfig?.max_new_tokens,
    'generation_config.max_new_tokens',
    modelId
  );
  const stabilityThreshold = readDiffusionGemmaPositiveInteger(
    generationConfig?.stability_threshold,
    'generation_config.stability_threshold',
    modelId
  );
  const padTokenId = readDiffusionGemmaTokenId(
    generationConfig?.pad_token_id ?? rawConfig?.pad_token_id,
    'generation_config.pad_token_id',
    modelId
  );
  const tMin = readDiffusionGemmaNonNegativeNumber(
    generationConfig?.t_min,
    'generation_config.t_min',
    modelId
  );
  const tMax = readDiffusionGemmaNonNegativeNumber(
    generationConfig?.t_max,
    'generation_config.t_max',
    modelId
  );
  if (tMax < tMin) {
    throw new Error(`DiffusionGemma model "${modelId}" requires generation_config.t_max >= t_min.`);
  }
  return {
    canvasLength,
    maxDenoisingSteps,
    maxNewTokens,
    tMin,
    tMax,
    entropyBound: readDiffusionGemmaEntropyBound(generationConfig, modelId),
    confidenceThreshold: readDiffusionGemmaNonNegativeNumber(
      generationConfig?.confidence_threshold,
      'generation_config.confidence_threshold',
      modelId
    ),
    stabilityThreshold,
    padTokenId,
    eosTokenIds: readDiffusionGemmaEosTokenIds(rawConfig, generationConfig, modelId),
    boiTokenId: readDiffusionGemmaNullableTokenId(rawConfig?.boi_token_id, 'config.boi_token_id', modelId),
    eoiTokenId: readDiffusionGemmaNullableTokenId(rawConfig?.eoi_token_id, 'config.eoi_token_id', modelId),
    imageTokenId: readDiffusionGemmaNullableTokenId(rawConfig?.image_token_id, 'config.image_token_id', modelId),
    selfConditioning: true,
    decoderCacheMode: 'encoder_kv_readonly_canvas_concat',
    router: {
      scaleHiddenStates: true,
      normalizeTopK: true,
      perExpertScale: true,
    },
  };
}

export function applyDiffusionGemmaInferenceContract(inference, rawConfig, generationConfig, modelId, resolvedModelType) {
  if (!resolveDiffusionGemmaConfig(rawConfig, resolvedModelType)) {
    return inference;
  }
  if (inference?.diffusionGemma && typeof inference.diffusionGemma === 'object') {
    return inference;
  }
  return {
    ...inference,
    diffusionGemma: resolveDiffusionGemmaInferenceContract(rawConfig, generationConfig, modelId),
  };
}

export function resolveConvertedAt(value) {
  if (value === undefined || value === null || value === '') {
    return new Date().toISOString();
  }
  if (typeof value !== 'string') {
    throw new Error('manifest convertedAt must be a string when provided.');
  }
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    throw new Error(`Invalid manifest convertedAt timestamp: "${value}"`);
  }
  return new Date(parsed).toISOString();
}

export function resolveManifestMultimodalConfig(rawConfig, manifestConfig = null) {
  const explicitVisionConfig = manifestConfig?.visionConfig;
  const explicitAudioConfig = manifestConfig?.audioConfig;
  const visionConfig = explicitVisionConfig ?? rawConfig?.vision_config ?? null;
  const audioConfig = explicitAudioConfig ?? rawConfig?.audio_config ?? null;
  return {
    vision_config: visionConfig ? cloneJsonValue(visionConfig) : null,
    audio_config: audioConfig ? cloneJsonValue(audioConfig) : null,
  };
}

export function createManifest(
  modelId,
  model,
  shards,
  tensorLocations,
  sourceOrOptions
) {
  if (!sourceOrOptions) {
    throw new Error('Missing manifest options');
  }
  const options = typeof sourceOrOptions === 'string' ? { source: sourceOrOptions } : sourceOrOptions ?? {};
  const source = options.source;
  if (!source) {
    throw new Error('Missing manifest source');
  }
  const resolvedModelType =
    options.modelType ??
    model.modelType ??
    model.config?.architectures?.[0] ??
    model.architecture;
  if (!resolvedModelType) {
    throw new Error('Missing modelType for manifest');
  }
  const isDiffusion = resolvedModelType === 'diffusion';
  const isEmbedding = resolvedModelType === 'embedding';
  const architecture = options.architecture ?? model.architecture ?? (
    isDiffusion ? 'diffusion' : extractArchitecture(model.config, model.ggufConfig)
  );
  const rawConfig = model.config || {};
  const generationConfig = model.generationConfig ?? null;
  const manifestPolicy = options.manifestConfig ?? null;
  const resolvedArchitecture = isDiffusion
    ? architecture
    : resolveIntermediateSizeFromTensors(architecture, model, tensorLocations, rawConfig, modelId);
  const moeConfig = isDiffusion
    ? null
    : resolveManifestMoEConfig(model, { ...options, modelId }, rawConfig, resolvedModelType);
  let inference = options.inference;
  if (!inference) {
    throw new Error('inference config is required — use a v1 conversion config');
  }
  inference = applyDiffusionGemmaInferenceContract(
    inference,
    rawConfig,
    generationConfig,
    modelId,
    resolvedModelType
  );

  const embeddingOutput = inferEmbeddingOutputConfig(tensorLocations);
  const hasExplicitEmbeddingPostprocessor = Object.prototype.hasOwnProperty.call(
    inference?.output ?? {},
    'embeddingPostprocessor'
  );
  const embeddingPostprocessor = hasExplicitEmbeddingPostprocessor
    ? inference?.output?.embeddingPostprocessor
    : (model.embeddingPostprocessor ?? null);
  if (embeddingOutput || hasExplicitEmbeddingPostprocessor || embeddingPostprocessor) {
    inference = {
      ...inference,
      output: {
        ...embeddingOutput,
        ...inference.output,
        embeddingPostprocessor,
      },
    };
  }

  const eosTokenId = options.eosTokenId !== undefined
    ? options.eosTokenId
    : isDiffusion || isEmbedding
      ? null
      : resolveEosTokenId({
          config: rawConfig,
          generationConfig,
          tokenizer: model.tokenizer ?? model.tokenizerConfig ?? null,
          tokenizerJson: model.tokenizerJson ?? null,
        });
  const resolvedQuantization = options.quantization ?? model.quantization;
  if (!resolvedQuantization) {
    throw new Error('Missing quantization for manifest');
  }
  const hashAlgorithm = options.hashAlgorithm;
  if (!hashAlgorithm) {
    throw new Error('Missing hashAlgorithm for manifest');
  }

  const isTextOnlyArtifact = options.textOnly === true;
  const multimodalConfig = isDiffusion || isTextOnlyArtifact
    ? { vision_config: null, audio_config: null }
    : resolveManifestMultimodalConfig(rawConfig, manifestPolicy);
  const manifestConfig = isDiffusion
    ? rawConfig
    : {
        ...(multimodalConfig.vision_config ? { vision_config: multimodalConfig.vision_config } : {}),
        ...(multimodalConfig.audio_config ? { audio_config: multimodalConfig.audio_config } : {}),
      };

  const manifest = {
    version: RDRR_VERSION,
    modelId,
    modelType: resolvedModelType,
    quantization: resolvedQuantization,
    quantizationInfo: options.quantizationInfo,
    ...(options.artifactIdentity ? { artifactIdentity: options.artifactIdentity } : {}),
    ...(options.weightsRef ? { weightsRef: options.weightsRef } : {}),
    architecture: resolvedArchitecture,
    moeConfig,
    inference,
    shards,
    tensors: tensorLocations,
    totalSize: shards.reduce((sum, s) => sum + s.size, 0),
    hashAlgorithm,
    eos_token_id: eosTokenId,
    ...(rawConfig.image_token_id !== undefined ? { image_token_id: rawConfig.image_token_id } : {}),
    ...(rawConfig.audio_token_id !== undefined ? { audio_token_id: rawConfig.audio_token_id } : {}),
    ...(rawConfig.video_token_id !== undefined ? { video_token_id: rawConfig.video_token_id } : {}),
    config: Object.keys(manifestConfig).length > 0 ? manifestConfig : undefined,
    conversion: options.conversionInfo,
    metadata: {
      source,
      convertedAt: resolveConvertedAt(
        options.convertedAt
        ?? options.conversionInfo?.convertedAt
      ),
    },
  };

  // Include tokenizer if available
  if (model.tokenizerJson) {
    manifest.tokenizer = buildBundledTokenizer(
      model.tokenizerJson,
      model.tokenizerConfig ?? null,
      rawConfig
    );
    manifest.metadata.hasTokenizer = true;
  } else {
    const tokenizer = buildSentencepieceTokenizer(
      model.tokenizerConfig ?? null,
      rawConfig,
      architecture,
      model.tokenizerModel ?? null
    );
    if (tokenizer) {
      manifest.tokenizer = tokenizer;
      manifest.metadata.hasTokenizer = true;
    }
  }

  return manifest;
}
