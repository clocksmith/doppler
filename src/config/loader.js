/**
 * Preset Loader
 *
 * Loads and merges model family presets with manifest config.
 * Implements config-as-code pattern: JSON presets, not if-statements.
 *
 * @module config/loader
 */

import { DEFAULT_LOADING_CONFIG } from './schema/index.js';

// Import presets statically for bundling
// In a real implementation, these would be loaded dynamically or bundled
import transformerPreset from './presets/models/transformer.json' with { type: 'json' };
import gemma2Preset from './presets/models/gemma2.json' with { type: 'json' };
import gemma3Preset from './presets/models/gemma3.json' with { type: 'json' };
import functiongemmaPreset from './presets/models/functiongemma.json' with { type: 'json' };
import llama3Preset from './presets/models/llama3.json' with { type: 'json' };
import mixtralPreset from './presets/models/mixtral.json' with { type: 'json' };
import deepseekPreset from './presets/models/deepseek.json' with { type: 'json' };
import mambaPreset from './presets/models/mamba.json' with { type: 'json' };
import qwen3Preset from './presets/models/qwen3.json' with { type: 'json' };
import kimiK2Preset from './presets/models/kimi-k2.json' with { type: 'json' };

// =============================================================================
// Preset Registry
// =============================================================================

/** @type {Record<string, import('./schema/index.js').PresetSchema>} */
export const PRESET_REGISTRY = {
  transformer: /** @type {import('./schema/index.js').PresetSchema} */ (transformerPreset),
  gemma2: /** @type {import('./schema/index.js').PresetSchema} */ (gemma2Preset),
  gemma3: /** @type {import('./schema/index.js').PresetSchema} */ (gemma3Preset),
  functiongemma: /** @type {import('./schema/index.js').PresetSchema} */ (functiongemmaPreset),
  llama3: /** @type {import('./schema/index.js').PresetSchema} */ (llama3Preset),
  mixtral: /** @type {import('./schema/index.js').PresetSchema} */ (mixtralPreset),
  deepseek: /** @type {import('./schema/index.js').PresetSchema} */ (deepseekPreset),
  mamba: /** @type {import('./schema/index.js').PresetSchema} */ (mambaPreset),
  qwen3: /** @type {import('./schema/index.js').PresetSchema} */ (qwen3Preset),
  kimi_k2: /** @type {import('./schema/index.js').PresetSchema} */ (kimiK2Preset),
};

// =============================================================================
// Preset Loading
// =============================================================================

/**
 * Get a preset by ID, with inheritance resolution.
 * @param {string} id
 * @returns {import('./schema/index.js').PresetSchema | null}
 */
export function getPreset(id) {
  return PRESET_REGISTRY[id] || null;
}

/**
 * List all available preset IDs.
 * @returns {string[]}
 */
export function listPresets() {
  return Object.keys(PRESET_REGISTRY);
}

/**
 * Resolve a preset with its parent chain merged.
 * Child values override parent values.
 * @param {string} id
 * @returns {import('./schema/index.js').PresetSchema}
 */
export function resolvePreset(id) {
  const preset = getPreset(id);
  if (!preset) {
    throw new Error(`Unknown preset: ${id}`);
  }

  // If no parent, return as-is
  if (!preset.extends) {
    return preset;
  }

  // Recursively resolve parent
  const parent = resolvePreset(preset.extends);

  // Deep merge parent and child
  return deepMergePresets(parent, preset);
}

// =============================================================================
// Model Detection
// =============================================================================

/**
 * Preset detection order - specific presets first, generic last.
 * This ensures e.g. "gemma2" is checked before "transformer".
 * @type {string[]}
 */
const PRESET_DETECTION_ORDER = [
  // Most specific first (model variants)
  'functiongemma',
  // Model families (check more specific patterns first)
  'gemma2',
  'gemma3',
  'llama3',
  'qwen3',
  'kimi_k2',
  'deepseek',  // Before mixtral (deepseek extends mixtral)
  'mixtral',
  'mamba',
  // Most generic last
  'transformer',
];

/**
 * Detect the best preset for a model based on its config.
 * Checks presets in order of specificity (most specific first).
 * @param {import('./schema/index.js').RawModelConfigSchema} config
 * @param {string} [architecture]
 * @returns {string}
 */
export function detectPreset(
  config,
  architecture
) {
  const archLower = (architecture || '').toLowerCase();
  const modelType = (config.model_type || '').toLowerCase();

  // Check presets in deterministic order (specific → generic)
  for (const id of PRESET_DETECTION_ORDER) {
    const preset = PRESET_REGISTRY[id];
    if (!preset?.detection) continue;

    // Check architecture patterns
    if (preset.detection.architecturePatterns) {
      for (const pattern of preset.detection.architecturePatterns) {
        if (archLower.includes(pattern.toLowerCase())) {
          return id;
        }
      }
    }

    // Check model type patterns
    if (preset.detection.modelTypePatterns) {
      for (const pattern of preset.detection.modelTypePatterns) {
        if (modelType.includes(pattern.toLowerCase())) {
          return id;
        }
      }
    }

    // Check config patterns
    if (preset.detection.configPatterns) {
      let allMatch = true;
      for (const [key, value] of Object.entries(preset.detection.configPatterns)) {
        if (config[key] !== value) {
          allMatch = false;
          break;
        }
      }
      if (allMatch && Object.keys(preset.detection.configPatterns).length > 0) {
        return id;
      }
    }
  }

  // Default to transformer
  return 'transformer';
}

// =============================================================================
// Config Resolution
// =============================================================================

/**
 * Build a fully resolved config by merging:
 * 1. Base preset (resolved with inheritance)
 * 2. Manifest config overrides
 * 3. Architecture from manifest
 * @param {import('./schema/index.js').ManifestSchema} manifest
 * @param {string} [presetId]
 * @returns {import('./schema/index.js').ResolvedConfigSchema}
 */
export function resolveConfig(
  manifest,
  presetId
) {
  // Detect or use provided preset
  const id = presetId || detectPreset(
    /** @type {import('./schema/index.js').RawModelConfigSchema} */ (manifest.config || {}),
    manifest.modelType
  );

  // Get resolved preset
  const preset = resolvePreset(id);

  // Extract architecture from manifest
  const manifestArch = typeof manifest.architecture === 'object'
    ? manifest.architecture
    : extractArchitectureFromConfig(manifest.config || {});

  // Merge architecture: preset defaults + manifest values
  // Note: Uses nullish coalesce (??) so null values fall through to next level.
  // This means explicit null in manifest = "use preset/default".
  const presetArch = preset.architecture || {};
  /** @type {import('./schema/index.js').ArchitectureSchema} */
  const architecture = {
    numLayers: manifestArch.numLayers ?? presetArch.numLayers ?? 32,
    hiddenSize: manifestArch.hiddenSize ?? presetArch.hiddenSize ?? 4096,
    intermediateSize: manifestArch.intermediateSize ?? presetArch.intermediateSize ?? 11008,
    numAttentionHeads: manifestArch.numAttentionHeads ?? presetArch.numAttentionHeads ?? 32,
    numKeyValueHeads: manifestArch.numKeyValueHeads ?? presetArch.numKeyValueHeads ?? 32,
    headDim: manifestArch.headDim ?? presetArch.headDim ?? 128,
    vocabSize: manifestArch.vocabSize ?? presetArch.vocabSize ?? 32000,
    maxSeqLen: manifestArch.maxSeqLen ?? presetArch.maxSeqLen ?? 2048,
    ropeTheta: manifestArch.ropeTheta ?? presetArch.ropeTheta,
    rmsNormEps: manifestArch.rmsNormEps ?? presetArch.rmsNormEps,
  };

  // Merge inference config
  // Note: Uses object spread, so explicit null in manifest/preset OVERRIDES base.
  // This differs from architecture (which uses ?? and ignores null).
  // Rationale: null values in inference (e.g., slidingWindow: null) mean "disabled".
  const baseInference = getDefaultInferenceConfig();
  const presetInference = preset.inference || {};
  const manifestInference = extractInferenceFromConfig(manifest.config || {});

  /** @type {Required<import('./schema/index.js').InferenceConfigSchema>} */
  const inference = {
    attention: {
      ...baseInference.attention,
      ...presetInference.attention,
      ...manifestInference.attention,
    },
    normalization: {
      ...baseInference.normalization,
      ...presetInference.normalization,
    },
    ffn: {
      ...baseInference.ffn,
      ...presetInference.ffn,
    },
    output: {
      ...baseInference.output,
      ...presetInference.output,
      ...manifestInference.output,
    },
    layerPattern: presetInference.layerPattern ?? baseInference.layerPattern,
    rope: {
      ...baseInference.rope,
      ...presetInference.rope,
      ...manifestInference.rope,
    },
    pipeline: manifestInference.pipeline ?? presetInference.pipeline ?? baseInference.pipeline,
    chatTemplate: {
      ...baseInference.chatTemplate,
      ...presetInference.chatTemplate,
    },
    kernelPath: presetInference.kernelPath ?? baseInference.kernelPath,
  };

  // Merge tokenizer config
  /** @type {import('./schema/index.js').TokenizerConfigSchema} */
  const tokenizer = {
    ...preset.tokenizer,
    ...extractTokenizerFromManifest(manifest),
  };

  // Sampling defaults
  const sampling = preset.sampling || {
    temperature: 1.0,
    topK: 50,
    topP: 0.95,
    repetitionPenalty: 1.0,
    maxTokens: 2048,
  };

  // Merge loading config: defaults + preset overrides
  const loading = mergeLoadingConfig(preset.loading);

  return {
    preset: id,
    modelType: preset.modelType || manifest.modelType || 'transformer',
    architecture,
    inference,
    tokenizer,
    sampling,
    loading,
  };
}

// =============================================================================
// Config Extraction Helpers
// =============================================================================

/**
 * Extract architecture from raw config (HF or GGUF style).
 * @param {Record<string, unknown>} config
 * @returns {Partial<import('./schema/index.js').ArchitectureSchema>}
 */
function extractArchitectureFromConfig(config) {
  return {
    numLayers: /** @type {number | undefined} */ (config.num_hidden_layers ?? config.n_layer ?? config.blockCount),
    hiddenSize: /** @type {number | undefined} */ (config.hidden_size ?? config.n_embd ?? config.embeddingLength),
    intermediateSize: /** @type {number | undefined} */ (config.intermediate_size ?? config.n_inner ?? config.feedForwardLength),
    numAttentionHeads: /** @type {number | undefined} */ (config.num_attention_heads ?? config.n_head ?? config.attentionHeadCount),
    numKeyValueHeads: /** @type {number | undefined} */ (config.num_key_value_heads ?? config.attentionHeadCountKV),
    headDim: /** @type {number | undefined} */ (config.head_dim),
    vocabSize: /** @type {number | undefined} */ (config.vocab_size ?? config.vocabSize),
    maxSeqLen: /** @type {number | undefined} */ (config.max_position_embeddings ?? config.n_positions ?? config.contextLength),
    ropeTheta: /** @type {number | undefined} */ (config.rope_theta ?? config.ropeFreqBase),
    rmsNormEps: /** @type {number | undefined} */ (config.rms_norm_eps ?? config.attentionLayerNormRMSEpsilon),
  };
}

/**
 * Extract inference config from raw config.
 * @param {Record<string, unknown>} config
 * @returns {Partial<import('./schema/index.js').InferenceConfigSchema>}
 */
function extractInferenceFromConfig(config) {
  return {
    attention: {
      slidingWindow: /** @type {number | null | undefined} */ (config.sliding_window),
      attnLogitSoftcapping: /** @type {number | null | undefined} */ (config.attn_logit_softcapping),
    },
    output: {
      finalLogitSoftcapping: /** @type {number | null | undefined} */ (config.final_logit_softcapping),
      tieWordEmbeddings: /** @type {boolean | undefined} */ (config.tie_word_embeddings),
    },
    pipeline: /** @type {import('./schema/index.js').InferenceConfigSchema['pipeline']} */ (config.pipeline),
    rope: {
      ropeTheta: /** @type {number | undefined} */ (config.rope_theta ?? config.ropeFreqBase),
      ropeScalingType: /** @type {'linear' | 'dynamic' | 'yarn' | null | undefined} */ (config.rope_scaling_type),
      ropeScalingFactor: /** @type {number | undefined} */ (config.rope_scaling_factor),
    },
  };
}

/**
 * Extract tokenizer config from manifest.
 * @param {import('./schema/index.js').ManifestSchema} manifest
 * @returns {Partial<import('./schema/index.js').TokenizerConfigSchema>}
 */
function extractTokenizerFromManifest(manifest) {
  if (!manifest.tokenizer) return {};

  return {
    // Could be extended to parse tokenizer.json content
  };
}

/**
 * Get default inference config.
 * @returns {Required<import('./schema/index.js').InferenceConfigSchema>}
 */
function getDefaultInferenceConfig() {
  return {
    attention: {
      slidingWindow: null,
      attnLogitSoftcapping: null,
      queryKeyNorm: false,
      ropeScalingType: null,
      ropeScalingFactor: 1.0,
    },
    normalization: {
      rmsNormWeightOffset: false,
      rmsNormEps: 1e-5,
      postAttentionNorm: false,
      preFeedforwardNorm: false,
      postFeedforwardNorm: false,
    },
    ffn: {
      activation: 'silu',
      gatedFFN: true,
      fusedGateUp: false,
    },
    output: {
      finalLogitSoftcapping: null,
      tieWordEmbeddings: false,
    },
    layerPattern: {
      type: 'all_attention',
    },
    rope: {
      ropeTheta: 10000,
      ropeLocalTheta: undefined,
      ropeScalingType: null,
      ropeScalingFactor: 1.0,
      yarnBetaFast: 32,
      yarnBetaSlow: 1,
      yarnOriginalMaxPos: 4096,
    },
    pipeline: null,
    chatTemplate: {
      type: null,
    },
    kernelPath: undefined,
  };
}

// =============================================================================
// Loading Config Merge
// =============================================================================

/**
 * Merge loading config with defaults.
 * Preset values override defaults.
 * @param {Partial<import('./schema/index.js').LoadingConfigSchema> | undefined} presetLoading
 * @returns {import('./schema/index.js').LoadingConfigSchema}
 */
function mergeLoadingConfig(presetLoading) {
  if (!presetLoading) {
    return DEFAULT_LOADING_CONFIG;
  }

  return {
    shardCache: {
      ...DEFAULT_LOADING_CONFIG.shardCache,
      ...presetLoading.shardCache,
    },
    memoryManagement: {
      ...DEFAULT_LOADING_CONFIG.memoryManagement,
      ...presetLoading.memoryManagement,
    },
    opfsPath: {
      ...DEFAULT_LOADING_CONFIG.opfsPath,
      ...presetLoading.opfsPath,
    },
    expertCache: {
      ...DEFAULT_LOADING_CONFIG.expertCache,
      ...presetLoading.expertCache,
    },
  };
}

// =============================================================================
// Deep Merge Utilities
// =============================================================================

/**
 * Deep merge two presets. Child values override parent.
 * @param {import('./schema/index.js').PresetSchema} parent
 * @param {import('./schema/index.js').PresetSchema} child
 * @returns {import('./schema/index.js').PresetSchema}
 */
function deepMergePresets(parent, child) {
  return {
    id: child.id,
    name: child.name ?? parent.name,
    extends: undefined, // Already resolved
    modelType: child.modelType ?? parent.modelType,
    architecture: mergePartial(parent.architecture, child.architecture),
    inference: mergeInference(parent.inference, child.inference),
    tokenizer: mergePartial(parent.tokenizer, child.tokenizer),
    sampling: mergePartial(parent.sampling, child.sampling),
    tensorPatterns: mergeTensorPatterns(parent.tensorPatterns, child.tensorPatterns),
    detection: child.detection ?? parent.detection,
    loading: mergePartial(parent.loading, child.loading),
  };
}

/**
 * Deep merge two objects. Child values override parent.
 * Handles nested objects recursively.
 * @template {Record<string, unknown>} T
 * @param {T} parent
 * @param {Partial<T>} child
 * @returns {T}
 */
function deepMerge(parent, child) {
  const result = /** @type {T} */ ({ ...parent });

  for (const key of /** @type {Array<keyof T>} */ (Object.keys(child))) {
    const childVal = child[key];
    const parentVal = parent[key];

    if (childVal === undefined) {
      continue;
    }

    if (
      childVal !== null &&
      typeof childVal === 'object' &&
      !Array.isArray(childVal) &&
      parentVal !== null &&
      typeof parentVal === 'object' &&
      !Array.isArray(parentVal)
    ) {
      // Recursively merge nested objects
      result[key] = /** @type {T[keyof T]} */ (deepMerge(
        /** @type {Record<string, unknown>} */ (parentVal),
        /** @type {Record<string, unknown>} */ (childVal)
      ));
    } else {
      // Override with child value
      result[key] = /** @type {T[keyof T]} */ (childVal);
    }
  }

  return result;
}

/**
 * Merge partial objects, child overrides parent.
 * Uses deep merge for nested objects.
 * @template {object} T
 * @param {T | undefined} parent
 * @param {T | undefined} child
 * @returns {T | undefined}
 */
function mergePartial(parent, child) {
  if (!parent && !child) return undefined;
  if (!parent) return child;
  if (!child) return parent;
  return /** @type {T} */ (deepMerge(
    /** @type {Record<string, unknown>} */ (parent),
    /** @type {Record<string, unknown>} */ (child)
  ));
}

/**
 * Merge inference config with nested objects.
 * @param {import('./schema/index.js').InferenceConfigSchema | undefined} parent
 * @param {import('./schema/index.js').InferenceConfigSchema | undefined} child
 * @returns {import('./schema/index.js').InferenceConfigSchema | undefined}
 */
function mergeInference(parent, child) {
  if (!parent && !child) return undefined;
  if (!parent) return child;
  if (!child) return parent;

  return {
    attention: mergePartial(parent.attention, child.attention),
    normalization: mergePartial(parent.normalization, child.normalization),
    ffn: mergePartial(parent.ffn, child.ffn),
    output: mergePartial(parent.output, child.output),
    layerPattern: child.layerPattern ?? parent.layerPattern,
    rope: mergePartial(parent.rope, child.rope),
    pipeline: child.pipeline ?? parent.pipeline,
    chatTemplate: mergePartial(parent.chatTemplate, child.chatTemplate),
    kernelPath: child.kernelPath ?? parent.kernelPath,
  };
}

/**
 * Merge tensor patterns with nested objects.
 * @param {import('./schema/index.js').PresetSchema['tensorPatterns']} parent
 * @param {import('./schema/index.js').PresetSchema['tensorPatterns']} child
 * @returns {import('./schema/index.js').PresetSchema['tensorPatterns']}
 */
function mergeTensorPatterns(parent, child) {
  if (!parent && !child) return undefined;
  if (!parent) return child;
  if (!child) return parent;

  return {
    embedding: child.embedding ?? parent.embedding,
    lmHead: child.lmHead ?? parent.lmHead,
    layer: child.layer ?? parent.layer,
    attention: mergePartial(parent.attention, child.attention),
    ffn: mergePartial(parent.ffn, child.ffn),
    norm: mergePartial(parent.norm, child.norm),
  };
}
