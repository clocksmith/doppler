/**
 * Preset Loader
 *
 * Loads and merges model family presets with manifest config.
 * Implements config-as-code pattern: JSON presets, not if-statements.
 *
 * @module config/loader
 */

import type {
  PresetSchema,
  ResolvedConfigSchema,
  ArchitectureSchema,
  InferenceConfigSchema,
  TokenizerConfigSchema,
  ManifestSchema,
  RawModelConfigSchema,
  LoadingConfigSchema,
} from './schema/index.js';
import { DEFAULT_LOADING_CONFIG } from './schema/index.js';

// =============================================================================
// Preset Registry
// =============================================================================

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

/** Registry of all available presets */
const PRESET_REGISTRY: Record<string, PresetSchema> = {
  transformer: transformerPreset as PresetSchema,
  gemma2: gemma2Preset as PresetSchema,
  gemma3: gemma3Preset as PresetSchema,
  functiongemma: functiongemmaPreset as PresetSchema,
  llama3: llama3Preset as PresetSchema,
  mixtral: mixtralPreset as PresetSchema,
  deepseek: deepseekPreset as PresetSchema,
  mamba: mambaPreset as PresetSchema,
  qwen3: qwen3Preset as PresetSchema,
  kimi_k2: kimiK2Preset as PresetSchema,
};

// =============================================================================
// Preset Loading
// =============================================================================

/**
 * Get a preset by ID, with inheritance resolution.
 */
export function getPreset(id: string): PresetSchema | null {
  return PRESET_REGISTRY[id] || null;
}

/**
 * List all available preset IDs.
 */
export function listPresets(): string[] {
  return Object.keys(PRESET_REGISTRY);
}

/**
 * Resolve a preset with its parent chain merged.
 * Child values override parent values.
 */
export function resolvePreset(id: string): PresetSchema {
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
 */
export function detectPreset(
  config: RawModelConfigSchema,
  architecture?: string
): string {
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
 */
export function resolveConfig(
  manifest: ManifestSchema,
  presetId?: string
): ResolvedConfigSchema {
  // Detect or use provided preset
  const id = presetId || detectPreset(
    (manifest.config || {}) as RawModelConfigSchema,
    manifest.modelType
  );

  // Get resolved preset
  const preset = resolvePreset(id);

  // Extract architecture from manifest
  const manifestArch = typeof manifest.architecture === 'object'
    ? manifest.architecture
    : extractArchitectureFromConfig(manifest.config || {});

  // Merge architecture: preset defaults + manifest values
  const presetArch = preset.architecture || {};
  const architecture: ArchitectureSchema = {
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
  const baseInference = getDefaultInferenceConfig();
  const presetInference = preset.inference || {};
  const manifestInference = extractInferenceFromConfig(manifest.config || {});

  const inference: Required<InferenceConfigSchema> = {
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
  };

  // Merge tokenizer config
  const tokenizer: TokenizerConfigSchema = {
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
 */
function extractArchitectureFromConfig(
  config: Record<string, unknown>
): Partial<ArchitectureSchema> {
  return {
    numLayers: (config.num_hidden_layers ?? config.n_layer ?? config.blockCount) as number | undefined,
    hiddenSize: (config.hidden_size ?? config.n_embd ?? config.embeddingLength) as number | undefined,
    intermediateSize: (config.intermediate_size ?? config.n_inner ?? config.feedForwardLength) as number | undefined,
    numAttentionHeads: (config.num_attention_heads ?? config.n_head ?? config.attentionHeadCount) as number | undefined,
    numKeyValueHeads: (config.num_key_value_heads ?? config.attentionHeadCountKV) as number | undefined,
    headDim: config.head_dim as number | undefined,
    vocabSize: (config.vocab_size ?? config.vocabSize) as number | undefined,
    maxSeqLen: (config.max_position_embeddings ?? config.n_positions ?? config.contextLength) as number | undefined,
    ropeTheta: (config.rope_theta ?? config.ropeFreqBase) as number | undefined,
    rmsNormEps: (config.rms_norm_eps ?? config.attentionLayerNormRMSEpsilon) as number | undefined,
  };
}

/**
 * Extract inference config from raw config.
 */
function extractInferenceFromConfig(
  config: Record<string, unknown>
): Partial<InferenceConfigSchema> {
  return {
    attention: {
      slidingWindow: config.sliding_window as number | null | undefined,
      attnLogitSoftcapping: config.attn_logit_softcapping as number | null | undefined,
    },
    output: {
      finalLogitSoftcapping: config.final_logit_softcapping as number | null | undefined,
      tieWordEmbeddings: config.tie_word_embeddings as boolean | undefined,
    },
    pipeline: config.pipeline as InferenceConfigSchema['pipeline'],
    rope: {
      ropeTheta: (config.rope_theta ?? config.ropeFreqBase) as number | undefined,
      ropeScalingType: config.rope_scaling_type as 'linear' | 'dynamic' | 'yarn' | null | undefined,
      ropeScalingFactor: config.rope_scaling_factor as number | undefined,
    },
  };
}

/**
 * Extract tokenizer config from manifest.
 */
function extractTokenizerFromManifest(
  manifest: ManifestSchema
): Partial<TokenizerConfigSchema> {
  if (!manifest.tokenizer) return {};

  return {
    // Could be extended to parse tokenizer.json content
  };
}

/**
 * Get default inference config.
 */
function getDefaultInferenceConfig(): Required<InferenceConfigSchema> {
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
  };
}

// =============================================================================
// Loading Config Merge
// =============================================================================

/**
 * Merge loading config with defaults.
 * Preset values override defaults.
 */
function mergeLoadingConfig(
  presetLoading: Partial<LoadingConfigSchema> | undefined
): LoadingConfigSchema {
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
 */
function deepMergePresets(parent: PresetSchema, child: PresetSchema): PresetSchema {
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
 */
function deepMerge<T extends Record<string, unknown>>(
  parent: T,
  child: Partial<T>
): T {
  const result = { ...parent } as T;

  for (const key of Object.keys(child) as Array<keyof T>) {
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
      result[key] = deepMerge(
        parentVal as Record<string, unknown>,
        childVal as Record<string, unknown>
      ) as T[keyof T];
    } else {
      // Override with child value
      result[key] = childVal as T[keyof T];
    }
  }

  return result;
}

/**
 * Merge partial objects, child overrides parent.
 * Uses deep merge for nested objects.
 */
function mergePartial<T extends object>(
  parent: T | undefined,
  child: T | undefined
): T | undefined {
  if (!parent && !child) return undefined;
  if (!parent) return child;
  if (!child) return parent;
  return deepMerge(parent as Record<string, unknown>, child as Record<string, unknown>) as T;
}

/**
 * Merge inference config with nested objects.
 */
function mergeInference(
  parent: InferenceConfigSchema | undefined,
  child: InferenceConfigSchema | undefined
): InferenceConfigSchema | undefined {
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
  };
}

/**
 * Merge tensor patterns with nested objects.
 */
function mergeTensorPatterns(
  parent: PresetSchema['tensorPatterns'],
  child: PresetSchema['tensorPatterns']
): PresetSchema['tensorPatterns'] {
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

// =============================================================================
// Exports
// =============================================================================

export { PRESET_REGISTRY };
