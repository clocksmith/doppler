import {
  EXECUTION_V1_SCHEMA_ID,
  expandExecutionV1,
} from '../config/schema/index.js';
import {
  buildQuantizationInfo,
  resolveManifestQuantization,
  resolveModelId,
} from './quantization-info.js';
import { sanitizeModelId } from './core.js';
import { classifyTensorRole } from '../formats/rdrr/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';

function normalizeWeightDtype(dtype) {
  if (!dtype) return null;
  const lower = String(dtype).trim().toLowerCase();
  const upper = String(dtype).trim().toUpperCase();
  const normalized = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtypeAlias', {
    dtype: lower,
    fallback: upper,
  });
  return normalized ? normalized.toUpperCase() : null;
}

function findTensorDtypeByRole(tensors, targetRole) {
  for (const tensor of (tensors || [])) {
    const name = typeof tensor?.name === 'string' ? tensor.name : '';
    if (!name) continue;
    if (classifyTensorRole(name) === targetRole) {
      return tensor?.dtype ?? null;
    }
  }
  return null;
}

function hasAnyTensorPattern(tensors, patterns) {
  const names = (tensors || []).map((t) => String(t?.name || '').toLowerCase());
  return names.some((name) => patterns.some((pattern) => name.includes(pattern)));
}

export function inferSourceWeightQuantization(tensors) {
  if (!Array.isArray(tensors) || tensors.length === 0) {
    throw new Error(
      'Cannot infer source weight quantization: no tensors provided. ' +
      'Set converterConfig.quantization.weights explicitly.'
    );
  }
  const weightTensors = [];
  for (const tensor of tensors) {
    const name = typeof tensor?.name === 'string' ? tensor.name : '';
    if (!name.includes('.weight')) continue;
    const dtype = normalizeWeightDtype(tensor?.dtype);
    if (!dtype) continue;
    weightTensors.push({ name, dtype });
  }
  const dtypes = new Set(weightTensors.map((tensor) => tensor.dtype));
  if (dtypes.size === 0) {
    throw new Error(
      'Cannot infer source weight quantization: no recognizable weight dtypes found. ' +
      'Set converterConfig.quantization.weights explicitly.'
    );
  }
  if (dtypes.size > 1) {
    const detail = Array.from(dtypes)
      .sort()
      .map((dtype) => {
        const names = weightTensors
          .filter((tensor) => tensor.dtype === dtype)
          .slice(0, 2)
          .map((tensor) => tensor.name);
        return names.length > 0 ? `${dtype} (${names.join(', ')})` : dtype;
      })
      .join('; ');
    throw new Error(
      `Ambiguous source weight dtypes: ${Array.from(dtypes).sort().join(', ')}. ` +
      `Samples: ${detail}. Set converterConfig.quantization.weights to override.`
    );
  }
  if (dtypes.size === 1) {
    const only = [...dtypes][0];
    if (only === 'F32') return 'f32';
    if (only === 'F16') return 'f16';
  }
  if (dtypes.has('F32')) return 'f32';
  return 'f16';
}

function normalizeKernelDtype(dtype) {
  if (!dtype) return null;
  const lower = String(dtype).trim().toLowerCase();
  if (!lower) return null;
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromDtypeAlias', {
    dtype: lower,
    fallback: null,
  });
}

function isV1Config(converterConfig) {
  const exec = converterConfig?.execution;
  return exec && typeof exec === 'object' && exec.kernels && typeof exec.kernels === 'object';
}


function validateV1InferenceFields(inference, modelId) {
  const required = ['attention', 'normalization', 'ffn', 'rope', 'output', 'chatTemplate'];
  for (const field of required) {
    if (!inference?.[field] || typeof inference[field] !== 'object') {
      throw new Error(
        `Config for "${modelId}" is missing required inference.${field}. ` +
        'V1 configs must provide all inference fields explicitly (no preset fallback).'
      );
    }
  }
}


function resolveConversionPlanV1(options) {
  const rawConfig = options?.rawConfig || {};
  const tensors = Array.isArray(options?.tensors) ? options.tensors : [];
  const converterConfig = options.converterConfig;
  const inference = converterConfig.inference;
  const execution = converterConfig.execution;
  const sessionDefaults = converterConfig.sessionDefaults;

  if (!inference || typeof inference !== 'object') {
    throw new Error(
      'V1 config requires an explicit inference section with all model inference fields.'
    );
  }
  if (!execution?.kernels || !execution?.decode || !execution?.prefill) {
    throw new Error(
      'V1 config requires execution with kernels, decode, and prefill arrays.'
    );
  }
  if (!sessionDefaults || typeof sessionDefaults !== 'object') {
    throw new Error(
      'V1 config requires sessionDefaults with compute defaults and kvcache policy.'
    );
  }
  if (!execution.policies || typeof execution.policies !== 'object') {
    throw new Error(
      'V1 config requires execution.policies.'
    );
  }

  // Validate the execution graph expands correctly (fail fast on bad tuples/kernels)
  expandExecutionV1(execution);

  const modelId = converterConfig?.output?.modelBaseId ?? rawConfig?.model_id ?? 'unknown';
  validateV1InferenceFields(inference, modelId);

  const sourceQuantization = (
    options?.sourceQuantization
    ?? converterConfig?.quantization?.weights
    ?? inferSourceWeightQuantization(tensors)
  );
  const weightOverride = converterConfig?.quantization?.weights ?? null;
  const embedDtypeRaw = normalizeWeightDtype(findTensorDtypeByRole(tensors, 'embedding'));
  const lmHeadDtypeRaw = normalizeWeightDtype(findTensorDtypeByRole(tensors, 'lm_head'));
  const hasVision = hasAnyTensorPattern(tensors, ['vision_', 'vision_tower', 'vision_model', 'image_encoder', 'visual.']);
  const hasAudio = hasAnyTensorPattern(tensors, ['audio_', 'audio_encoder', 'whisper', 'wav2vec']);
  const hasProjector = hasAnyTensorPattern(tensors, ['multi_modal_projector', 'mm_projector', 'projector']);
  const quantizationInfo = buildQuantizationInfo(
    converterConfig,
    sourceQuantization,
    embedDtypeRaw,
    lmHeadDtypeRaw,
    hasVision,
    hasAudio,
    hasProjector,
    rawConfig
  );
  const manifestQuantization = resolveManifestQuantization(weightOverride, sourceQuantization);

  const modelType = converterConfig?.modelType ?? rawConfig?.model_type ?? 'transformer';

  // Build manifest inference directly from config (no preset resolution)
  const manifestInference = {
    schema: EXECUTION_V1_SCHEMA_ID,
    attention: inference.attention,
    normalization: inference.normalization,
    ffn: inference.ffn,
    rope: inference.rope,
    output: inference.output,
    layerPattern: inference.layerPattern ?? { type: 'uniform', globalPattern: null, period: null, offset: null, layerTypes: null },
    chatTemplate: inference.chatTemplate,
    pipeline: inference.pipeline ?? null,
    sessionDefaults,
    execution,
  };

  return {
    modelType,
    presetId: null,
    preset: null,
    sourceQuantization,
    quantizationInfo,
    manifestQuantization,
    manifestInference,
    headDim: options?.headDim ?? options?.architectureConfig?.headDim ?? null,
    executionVersion: 'v1',
  };
}


export function resolveConversionPlan(options) {
  const rawConfig = options?.rawConfig || {};
  const tensors = Array.isArray(options?.tensors) ? options.tensors : [];
  const tensorNames = options?.tensorNames ?? tensors.map((tensor) => tensor.name);
  const converterConfig = options?.converterConfig;
  if (converterConfig == null) {
    throw new Error(
      'resolveConversionPlan requires an explicit converterConfig. ' +
      'Provide a conversion config JSON (see tools/configs/conversion/ for examples).'
    );
  }

  // V1 config: explicit execution graph — the only supported converter path
  if (isV1Config(converterConfig)) {
    return resolveConversionPlanV1(options);
  }

  throw new Error(
    'converterConfig must have an execution.kernels object (v1 format). ' +
    'Legacy preset-based conversion (v0) is no longer supported. ' +
    'Use a v1 conversion config — see tools/configs/conversion/ for examples.'
  );
}

export function resolveConvertedModelId(options) {
  const explicitModelId = options?.explicitModelId ?? null;
  const converterConfig = options?.converterConfig ?? null;
  const detectedModelId = options?.detectedModelId ?? null;
  const quantizationInfo = options?.quantizationInfo ?? null;
  const fallbackModelId = options?.fallbackModelId ?? null;
  const sanitizeOnly = options?.sanitizeOnly === true;

  if (explicitModelId) {
    return sanitizeModelId(explicitModelId);
  }

  const baseModelId = (
    converterConfig?.output?.modelBaseId
    ?? detectedModelId
    ?? fallbackModelId
  );
  if (!baseModelId) return null;
  const resolved = sanitizeOnly
    ? baseModelId
    : resolveModelId(baseModelId, detectedModelId ?? baseModelId, quantizationInfo?.variantTag);
  return sanitizeModelId(resolved);
}
