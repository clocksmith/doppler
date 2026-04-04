import {
  createConverterConfig,
  DEFAULT_EXECUTION_V1_SESSION,
  DEFAULT_MANIFEST_INFERENCE,
} from '../config/schema/index.js';

const ZERO_DIGEST = 'sha256:' + '0'.repeat(64);

function cloneJsonValue(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function readRawConfigField(rawConfig, key) {
  if (!rawConfig || typeof rawConfig !== 'object') {
    return undefined;
  }
  if (rawConfig[key] !== undefined) {
    return rawConfig[key];
  }
  const textConfig = rawConfig.text_config;
  if (textConfig && typeof textConfig === 'object' && textConfig[key] !== undefined) {
    return textConfig[key];
  }
  return undefined;
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

export function createSourceRuntimeInference(rawConfig = null) {
  const inference = cloneJsonValue(DEFAULT_MANIFEST_INFERENCE);

  const rmsNormEps = asFinitePositiveNumber(readRawConfigField(rawConfig, 'rms_norm_eps'));
  if (rmsNormEps != null) {
    inference.normalization.rmsNormEps = rmsNormEps;
  }

  const ropeTheta = asFinitePositiveNumber(
    readRawConfigField(rawConfig, 'rope_theta') ?? readRawConfigField(rawConfig, 'rope_freq_base')
  );
  if (ropeTheta != null) {
    inference.rope.ropeTheta = ropeTheta;
  }

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

export function createSourceRuntimeConverterConfig(options = {}) {
  return createConverterConfig({
    quantization: options.quantization ?? undefined,
    output: {
      modelBaseId: options.modelId ?? null,
    },
    inference: createSourceRuntimeInference(options.rawConfig ?? null),
    session: createSourceRuntimeSession(),
    execution: createSourceRuntimeExecution(),
  });
}
