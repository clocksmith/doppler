import {
  createConverterConfig,
  EXECUTION_V1_SCHEMA_ID,
  DEFAULT_EXECUTION_V1_SESSION,
  DEFAULT_MANIFEST_INFERENCE,
} from '../config/schema/index.js';
import { buildRoPEConfig } from '../converter/rope-config.js';
import { cloneJsonValue } from '../utils/clone-json.js';

const ZERO_DIGEST = 'sha256:' + '0'.repeat(64);

function readRawConfigField(rawConfig, key) {
  if (!rawConfig || typeof rawConfig !== 'object') {
    return undefined;
  }
  const topLevelValue = rawConfig[key];
  const textConfig = rawConfig.text_config;
  if (topLevelValue !== undefined && topLevelValue !== null) {
    return topLevelValue;
  }
  if (textConfig && typeof textConfig === 'object' && textConfig[key] !== undefined) {
    return textConfig[key];
  }
  return topLevelValue;
}

function asFinitePositiveNumber(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return parsed;
}

function asOptionalBoolean(value) {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === 'boolean') {
    return value;
  }
  return null;
}

function normalizeActivation(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (normalized === 'gelu_pytorch_tanh' || normalized === 'gelu_new') {
    return 'gelu';
  }
  if (normalized === 'silu' || normalized === 'gelu' || normalized === 'relu' || normalized === 'swiglu') {
    return normalized;
  }
  return null;
}

function normalizeModelFamily(rawConfig) {
  const rawModelType = String(readRawConfigField(rawConfig, 'model_type') ?? rawConfig?.model_type ?? '').trim().toLowerCase();
  if (rawModelType === 'gemma3' || rawModelType === 'gemma3_text') {
    return 'gemma3';
  }
  if (rawModelType === 'gemma4' || rawModelType === 'gemma4_text') {
    return 'gemma4';
  }
  return rawModelType;
}

function applyFamilyDefaults(inference, rawConfig) {
  const family = normalizeModelFamily(rawConfig);
  if (family === 'gemma3') {
    inference.attention.queryPreAttnScalar = 256;
    inference.attention.queryKeyNorm = true;
    inference.attention.valueNorm = false;
    inference.normalization.rmsNormWeightOffset = true;
    inference.normalization.postAttentionNorm = true;
    inference.normalization.preFeedforwardNorm = true;
    inference.normalization.postFeedforwardNorm = true;
    inference.output.scaleEmbeddings = true;
    inference.chatTemplate.type = 'gemma';
    inference.chatTemplate.enabled = true;
    return family;
  }
  if (family === 'gemma4') {
    inference.attention.queryPreAttnScalar = 1;
    inference.attention.queryKeyNorm = true;
    inference.attention.valueNorm = true;
    inference.normalization.rmsNormWeightOffset = false;
    inference.normalization.postAttentionNorm = true;
    inference.normalization.preFeedforwardNorm = true;
    inference.normalization.postFeedforwardNorm = true;
    inference.output.scaleEmbeddings = true;
    inference.chatTemplate.type = 'gemma4';
    inference.chatTemplate.enabled = true;
    return family;
  }
  return family;
}

function applyExplicitAttentionConfig(inference, rawConfig) {
  const attentionBias = asOptionalBoolean(readRawConfigField(rawConfig, 'attention_bias'));
  if (attentionBias != null) {
    inference.attention.attentionBias = attentionBias;
  }

  const queryPreAttnScalar = asFinitePositiveNumber(readRawConfigField(rawConfig, 'query_pre_attn_scalar'));
  if (queryPreAttnScalar != null) {
    inference.attention.queryPreAttnScalar = queryPreAttnScalar;
  }

  const queryKeyNorm = asOptionalBoolean(readRawConfigField(rawConfig, 'query_key_norm'));
  if (queryKeyNorm != null) {
    inference.attention.queryKeyNorm = queryKeyNorm;
  }

  const valueNorm = asOptionalBoolean(readRawConfigField(rawConfig, 'value_norm'));
  if (valueNorm != null) {
    inference.attention.valueNorm = valueNorm;
  }

  const finalLogitSoftcapping = asFinitePositiveNumber(readRawConfigField(rawConfig, 'final_logit_softcapping'));
  if (finalLogitSoftcapping != null) {
    inference.output.finalLogitSoftcapping = finalLogitSoftcapping;
  }
}

function applyExplicitFfnConfig(inference, rawConfig) {
  const useDoubleWideMlp = asOptionalBoolean(readRawConfigField(rawConfig, 'use_double_wide_mlp'));
  if (useDoubleWideMlp != null) {
    inference.ffn.useDoubleWideMlp = useDoubleWideMlp;
  }
}

function applyLayerPatternConfig(inference, rawConfig) {
  const rawLayerTypes = readRawConfigField(rawConfig, 'layer_types');
  if (Array.isArray(rawLayerTypes) && rawLayerTypes.length > 0) {
    inference.layerPattern.type = 'custom';
    inference.layerPattern.globalPattern = null;
    inference.layerPattern.period = null;
    inference.layerPattern.offset = null;
    inference.layerPattern.layerTypes = [...rawLayerTypes];
    return;
  }

  const slidingWindowPattern = asFinitePositiveNumber(readRawConfigField(rawConfig, 'sliding_window_pattern'));
  if (slidingWindowPattern != null) {
    inference.layerPattern.type = 'every_n';
    inference.layerPattern.globalPattern = null;
    inference.layerPattern.period = Math.trunc(slidingWindowPattern);
    inference.layerPattern.offset = null;
    inference.layerPattern.layerTypes = null;
  }
}

function resolveSourceRuntimeVisionConfig(rawConfig) {
  const visionConfig = rawConfig?.vision_config;
  if (!visionConfig || typeof visionConfig !== 'object' || Array.isArray(visionConfig)) {
    return null;
  }
  const modelType = String(visionConfig.model_type ?? '').trim().toLowerCase();
  const visionArchitecture = String(visionConfig.vision_architecture ?? '').trim()
    || (modelType === 'gemma4_vision' ? 'gemma4' : '')
    || (modelType === 'qwen3_vl' || modelType === 'qwen3vl' ? 'qwen3vl' : '');
  return {
    ...cloneJsonValue(visionConfig),
    ...(visionArchitecture ? { vision_architecture: visionArchitecture } : {}),
  };
}

function resolveSourceRuntimeAudioConfig(rawConfig) {
  const audioConfig = rawConfig?.audio_config;
  if (!audioConfig || typeof audioConfig !== 'object' || Array.isArray(audioConfig)) {
    return null;
  }
  const modelType = String(audioConfig.model_type ?? '').trim().toLowerCase();
  const audioArchitecture = String(audioConfig.audio_architecture ?? '').trim()
    || (modelType === 'gemma4_audio' ? 'gemma4' : '');
  return {
    ...cloneJsonValue(audioConfig),
    ...(audioArchitecture ? { audio_architecture: audioArchitecture } : {}),
  };
}

export function createSourceRuntimeInference(rawConfig = null) {
  const inference = cloneJsonValue(DEFAULT_MANIFEST_INFERENCE);
  applyFamilyDefaults(inference, rawConfig);

  const rmsNormEps = asFinitePositiveNumber(readRawConfigField(rawConfig, 'rms_norm_eps'));
  if (rmsNormEps != null) {
    inference.normalization.rmsNormEps = rmsNormEps;
  }

  Object.assign(inference.rope, buildRoPEConfig(inference, rawConfig));

  const slidingWindow = readRawConfigField(rawConfig, 'sliding_window');
  if (slidingWindow === null) {
    inference.attention.slidingWindow = null;
  } else {
    const parsedSlidingWindow = asFinitePositiveNumber(slidingWindow);
    if (parsedSlidingWindow != null) {
      inference.attention.slidingWindow = Math.trunc(parsedSlidingWindow);
    }
  }

  const activation = normalizeActivation(
    readRawConfigField(rawConfig, 'hidden_act') ?? readRawConfigField(rawConfig, 'hidden_activation')
  );
  if (activation) {
    inference.ffn.activation = activation;
  }

  const tieWordEmbeddings = asOptionalBoolean(readRawConfigField(rawConfig, 'tie_word_embeddings'));
  if (tieWordEmbeddings != null) {
    inference.output.tieWordEmbeddings = tieWordEmbeddings;
  }

  const scaleEmbeddings = asOptionalBoolean(readRawConfigField(rawConfig, 'scale_embeddings'));
  if (scaleEmbeddings != null) {
    inference.output.scaleEmbeddings = scaleEmbeddings;
  }

  applyExplicitAttentionConfig(inference, rawConfig);
  applyExplicitFfnConfig(inference, rawConfig);
  applyLayerPatternConfig(inference, rawConfig);

  return inference;
}

export function createSourceRuntimeExecution() {
  return {
    kernels: {
      embed: {
        kernel: 'gather_f16.wgsl',
        entry: 'main',
        digest: ZERO_DIGEST,
      },
    },
    preLayer: [['embed', 'embed', 'embed_tokens']],
    decode: [],
    prefill: [],
    postLayer: [],
    policies: {
      unsupportedPrecision: 'error',
      dtypeTransition: 'require_cast_step',
      unresolvedKernel: 'error',
    },
  };
}

export function createSourceRuntimeSession() {
  return cloneJsonValue(DEFAULT_EXECUTION_V1_SESSION);
}

export function createSourceRuntimeManifestConfig(rawConfig = null) {
  return {
    hashAlgorithm: 'sha256',
    visionConfig: resolveSourceRuntimeVisionConfig(rawConfig),
    audioConfig: resolveSourceRuntimeAudioConfig(rawConfig),
  };
}

export function createSourceRuntimeManifestInference(rawConfig = null) {
  const inference = createSourceRuntimeInference(rawConfig);
  return {
    schema: EXECUTION_V1_SCHEMA_ID,
    attention: inference.attention,
    normalization: inference.normalization,
    ffn: inference.ffn,
    rope: inference.rope,
    output: inference.output,
    layerPattern: inference.layerPattern,
    chatTemplate: inference.chatTemplate,
    pipeline: inference.pipeline ?? null,
    session: createSourceRuntimeSession(),
    execution: createSourceRuntimeExecution(),
  };
}

export function createSourceRuntimeConverterConfig(options = {}) {
  return createConverterConfig({
    quantization: options.quantization ?? undefined,
    manifest: createSourceRuntimeManifestConfig(options.rawConfig ?? null),
    output: {
      modelBaseId: options.modelId ?? null,
    },
    inference: createSourceRuntimeInference(options.rawConfig ?? null),
    session: createSourceRuntimeSession(),
    execution: createSourceRuntimeExecution(),
  });
}
