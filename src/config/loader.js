import {
  DEFAULT_LOADING_CONFIG,
  DEFAULT_PRESET_INFERENCE_CONFIG,
  DEFAULT_SAMPLING_DEFAULTS,
} from './schema/index.js';

// Static imports keep presets bundled for browser use.
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
import gptOssPreset from './presets/models/gpt-oss.json' with { type: 'json' };

// =============================================================================
// Preset Registry
// =============================================================================

export const PRESET_REGISTRY = {
  transformer: transformerPreset,
  gemma2: gemma2Preset,
  gemma3: gemma3Preset,
  functiongemma: functiongemmaPreset,
  llama3: llama3Preset,
  mixtral: mixtralPreset,
  deepseek: deepseekPreset,
  mamba: mambaPreset,
  qwen3: qwen3Preset,
  kimi_k2: kimiK2Preset,
  gpt_oss: gptOssPreset,
};

// =============================================================================
// Preset Loading
// =============================================================================

export function getPreset(id) {
  return PRESET_REGISTRY[id] || null;
}

export function listPresets() {
  return Object.keys(PRESET_REGISTRY);
}

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

const PRESET_DETECTION_ORDER = [
  // Most specific first (model variants)
  'functiongemma',
  // Model families (check more specific patterns first)
  'gemma2',
  'gemma3',
  'llama3',
  'qwen3',
  'kimi_k2',
  'gpt_oss',
  'deepseek',  // Before mixtral (deepseek extends mixtral)
  'mixtral',
  'mamba',
  // Most generic last
  'transformer',
];

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

export function resolveConfig(
  manifest,
  presetId
) {
  // Detect or use provided preset
  const id = presetId || detectPreset(
    (manifest.config || {}),
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
  const baseInference = DEFAULT_PRESET_INFERENCE_CONFIG;
  const presetInference = preset.inference || {};
  const manifestInference = extractInferenceFromConfig(manifest.config || {});

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
  const tokenizer = {
    ...preset.tokenizer,
    ...extractTokenizerFromManifest(manifest),
  };

  // Sampling defaults
  const sampling = preset.sampling ?? {
    temperature: DEFAULT_SAMPLING_DEFAULTS.temperature,
    topK: DEFAULT_SAMPLING_DEFAULTS.topK,
    topP: DEFAULT_SAMPLING_DEFAULTS.topP,
    repetitionPenalty: DEFAULT_SAMPLING_DEFAULTS.repetitionPenalty,
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

function extractArchitectureFromConfig(config) {
  return {
    numLayers: config.num_hidden_layers ?? config.n_layer ?? config.blockCount,
    hiddenSize: config.hidden_size ?? config.n_embd ?? config.embeddingLength,
    intermediateSize: config.intermediate_size ?? config.n_inner ?? config.feedForwardLength,
    numAttentionHeads: config.num_attention_heads ?? config.n_head ?? config.attentionHeadCount,
    numKeyValueHeads: config.num_key_value_heads ?? config.attentionHeadCountKV,
    headDim: config.head_dim,
    vocabSize: config.vocab_size ?? config.vocabSize,
    maxSeqLen: config.max_position_embeddings ?? config.n_positions ?? config.contextLength,
    ropeTheta: config.rope_theta ?? config.ropeFreqBase,
    rmsNormEps: config.rms_norm_eps ?? config.attentionLayerNormRMSEpsilon,
  };
}

function extractInferenceFromConfig(config) {
  return {
    attention: {
      slidingWindow: config.sliding_window,
      attnLogitSoftcapping: config.attn_logit_softcapping,
    },
    output: {
      finalLogitSoftcapping: config.final_logit_softcapping,
      tieWordEmbeddings: config.tie_word_embeddings,
      scaleEmbeddings: config.scale_embeddings,
    },
    pipeline: config.pipeline,
    rope: {
      ropeTheta: config.rope_theta ?? config.ropeFreqBase,
      ropeScalingType: config.rope_scaling_type,
      ropeScalingFactor: config.rope_scaling_factor,
    },
  };
}

function extractTokenizerFromManifest(manifest) {
  if (!manifest.tokenizer) return {};

  return {
  };
}

// =============================================================================
// Loading Config Merge
// =============================================================================

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

function deepMerge(parent, child) {
  const result = { ...parent };

  for (const key of Object.keys(child)) {
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
      result[key] = deepMerge(parentVal, childVal);
    } else {
      // Override with child value
      result[key] = childVal;
    }
  }

  return result;
}

function mergePartial(parent, child) {
  if (!parent && !child) return undefined;
  if (!parent) return child;
  if (!child) return parent;
  return deepMerge(parent, child);
}

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
    kernelPaths: child.kernelPaths ?? parent.kernelPaths,
    kernelPath: child.kernelPath ?? parent.kernelPath,
  };
}

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
