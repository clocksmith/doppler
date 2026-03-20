import { resolveKernelPath, getKernelPathActivationDtype } from '../config/kernel-path-loader.js';
import { detectPreset, listPresets, resolvePreset } from '../config/loader.js';
import {
  DEFAULT_MANIFEST_INFERENCE,
  EXECUTION_V1_SCHEMA_ID,
  expandExecutionV1,
} from '../config/schema/index.js';
import { buildManifestInference } from './manifest-inference.js';
import {
  buildQuantizationInfo,
  resolveManifestQuantization,
  resolveModelId,
} from './quantization-info.js';
import { sanitizeModelId } from './core.js';
import { classifyTensorRole } from '../formats/rdrr/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { buildKernelRefFromKernelEntry, isKernelRefBoundToKernel } from '../config/kernels/kernel-ref.js';
import { mergeLayeredShallowObjects } from '../config/merge-helpers.js';

const KNOWN_MODEL_PRESETS = new Set(listPresets());
const CONVERSION_SUPPORTED_PRESETS = [...KNOWN_MODEL_PRESETS]
  .filter((presetId) => !['transformer', 'diffusion'].includes(presetId))
  .sort()
  .join(', ');
const EXECUTION_PHASES = new Set(['prefill', 'decode', 'both']);
const EXECUTION_SECTIONS = new Set(['preLayer', 'layer', 'postLayer', 'sampling']);
const SEMVER_PATTERN = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;
const SHA256_DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;
function isSemver(value) { return typeof value === 'string' && SEMVER_PATTERN.test(value); }
function isSha256Digest(value) { return typeof value === 'string' && SHA256_DIGEST_PATTERN.test(value); }

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

function normalizeKernelDtype(dtype) {
  if (!dtype) return null;
  const lower = String(dtype).trim().toLowerCase();
  if (!lower) return null;
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromDtypeAlias', {
    dtype: lower,
    fallback: null,
  });
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

function buildUnknownFamilyError(architectureHint, rawConfig, includePresetOverrideHint = false) {
  const modelType = rawConfig?.model_type ?? 'unknown';
  const overrideHint = includePresetOverrideHint
    ? `  2. Set converterConfig.presets.model to a known family (e.g., embeddinggemma)\n`
    : '';
  const createPresetHint = includePresetOverrideHint
    ? `  3. Create a custom preset in src/config/presets/models/\n`
    : `  2. Create a custom preset in src/config/presets/models/\n`;
  const issueHint = includePresetOverrideHint
    ? `  4. File an issue at https://github.com/clocksmith/doppler/issues\n\n`
    : `  3. File an issue at https://github.com/clocksmith/doppler/issues\n\n`;
  return new Error(
    `Unknown model family: architecture="${architectureHint || 'unknown'}", model_type="${modelType}"\n\n` +
    `DOPPLER requires a known model preset to generate correct inference config.\n` +
    `The manifest-first architecture does not support generic defaults.\n\n` +
    `Options:\n` +
    `  1. Wait for official support of this model family\n` +
    overrideHint +
    createPresetHint +
    issueHint +
    `Supported model families: ${CONVERSION_SUPPORTED_PRESETS}`
  );
}

function isLikelyEmbeddingGemma(rawConfig, architectureHint) {
  const arch = String(architectureHint || '').toLowerCase();
  if (arch.includes('embeddinggemma')) {
    return true;
  }

  const modelType = String(
    rawConfig?.model_type
    ?? rawConfig?.text_config?.model_type
    ?? ''
  ).toLowerCase();
  const useBidirectional = (
    rawConfig?.use_bidirectional_attention
    ?? rawConfig?.text_config?.use_bidirectional_attention
  ) === true;
  const isEmbeddingModelType = modelType.includes('embeddinggemma');

  if (arch.includes('gemma3textmodel')) {
    return useBidirectional || isEmbeddingModelType;
  }

  return (
    useBidirectional && (modelType === 'gemma3_text' || modelType === 'gemma3text')
  ) || isEmbeddingModelType;
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

export function validateDefaultKernelPath(inference, context = {}) {
  if (!inference?.defaultKernelPath) return;
  let resolvedKernelPath;
  try {
    resolvedKernelPath = resolveKernelPath(inference.defaultKernelPath);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const presetId = context?.presetId ?? 'unknown';
    const quantizationInfo = context?.quantizationInfo ?? null;
    throw new Error(
      `Invalid defaultKernelPath "${inference.defaultKernelPath}" for preset "${presetId}" ` +
      `(weights=${quantizationInfo?.weights ?? 'unknown'}, compute=${quantizationInfo?.compute ?? 'default'}, ` +
      `q4kLayout=${quantizationInfo?.layout ?? 'row'}): ${message}`
    );
  }

  const quantizationInfo = context?.quantizationInfo ?? null;
  const expectedComputeDtype = normalizeKernelDtype(quantizationInfo?.compute);
  const kernelActivationDtype = normalizeKernelDtype(
    getKernelPathActivationDtype(resolvedKernelPath)
  );
  if (
    expectedComputeDtype
    && kernelActivationDtype
    && expectedComputeDtype !== kernelActivationDtype
  ) {
    const presetId = context?.presetId ?? 'unknown';
    throw new Error(
      `Invalid defaultKernelPath "${inference.defaultKernelPath}" for preset "${presetId}" ` +
      `(weights=${quantizationInfo?.weights ?? 'unknown'}, compute=${expectedComputeDtype}, ` +
      `q4kLayout=${quantizationInfo?.layout ?? 'row'}): ` +
      `kernel activation dtype "${kernelActivationDtype}" is incompatible with compute "${expectedComputeDtype}".`
    );
  }
}

function readConverterKernelPathOverride(converterConfig) {
  const raw = converterConfig?.inference?.defaultKernelPath;
  if (raw == null) return null;
  if (typeof raw !== 'string') {
    throw new Error('converterConfig.inference.defaultKernelPath must be a string when provided.');
  }
  const trimmed = raw.trim();
  return trimmed || null;
}

function cloneJson(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}



function readConverterSessionDefaultsOverride(converterConfig) {
  const raw = converterConfig?.inference?.sessionDefaults;
  if (raw == null) return null;
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(
      'converterConfig.inference.sessionDefaults must be an object when provided.'
    );
  }
  return cloneJson(raw);
}

function assertNonExecutionSessionDefaults(manifestInference) {
  const sessionDefaults = manifestInference?.sessionDefaults;
  if (sessionDefaults == null) {
    return;
  }
  if (typeof sessionDefaults !== 'object' || Array.isArray(sessionDefaults)) {
    throw new Error(
      'converterConfig.inference.sessionDefaults must resolve to an object for non-execution manifests.'
    );
  }
  const keys = Object.keys(sessionDefaults);
  const invalidKeys = keys.filter((key) => key !== 'decodeLoop');
  if (invalidKeys.length > 0) {
    throw new Error(
      'converterConfig.inference.sessionDefaults may only set decodeLoop unless ' +
      'converterConfig.inference.execution is present.'
    );
  }
}

function readConverterExecutionOverride(converterConfig) {
  const raw = converterConfig?.inference?.execution;
  if (raw == null) return null;
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('converterConfig.inference.execution must be an object when provided.');
  }
  const steps = raw.steps;
  if (!Array.isArray(steps)) {
    throw new Error('converterConfig.inference.execution.steps must be an array.');
  }
  if (raw.policies != null && (typeof raw.policies !== 'object' || Array.isArray(raw.policies))) {
    throw new Error('converterConfig.inference.execution.policies must be an object when provided.');
  }
  validateConverterExecutionSteps(steps);
  return cloneJson(raw);
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value.trim();
}

function assertOptionalDtype(value, label) {
  if (value == null) return;
  const normalized = String(value).trim().toLowerCase();
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(`${label} must be "f16" or "f32".`);
  }
}

function validateConverterExecutionSteps(steps) {
  const ids = new Set();
  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    const prefix = `converterConfig.inference.execution.steps[${index}]`;
    if (!step || typeof step !== 'object' || Array.isArray(step)) {
      throw new Error(`${prefix} must be an object.`);
    }

    const id = assertString(step.id, `${prefix}.id`);
    if (ids.has(id)) {
      throw new Error(`${prefix}.id "${id}" is duplicated.`);
    }
    ids.add(id);

    assertString(step.op, `${prefix}.op`);
    const phase = assertString(step.phase, `${prefix}.phase`).toLowerCase();
    if (!EXECUTION_PHASES.has(phase)) {
      throw new Error(`${prefix}.phase must be prefill|decode|both.`);
    }
    const section = assertString(step.section, `${prefix}.section`);
    if (!EXECUTION_SECTIONS.has(section)) {
      throw new Error(`${prefix}.section must be preLayer|layer|postLayer|sampling.`);
    }
    assertString(step.src, `${prefix}.src`);
    assertString(step.dst, `${prefix}.dst`);

    if (step.layers !== 'all' && !Array.isArray(step.layers)) {
      throw new Error(`${prefix}.layers must be "all" or number[].`);
    }
    if (Array.isArray(step.layers)) {
      for (const layer of step.layers) {
        if (!Number.isInteger(layer) || layer < 0) {
          throw new Error(`${prefix}.layers must contain non-negative integers.`);
        }
      }
    }

    if (step.op === 'cast') {
      assertOptionalDtype(step.fromDtype, `${prefix}.fromDtype`);
      assertOptionalDtype(step.toDtype, `${prefix}.toDtype`);
      if (step.toDtype == null) {
        throw new Error(`${prefix}.toDtype is required for cast steps.`);
      }
      continue;
    }

    const kernel = assertString(step.kernel, `${prefix}.kernel`);
    const entry = String(step.entry ?? 'main').trim() || 'main';
    if (!step.kernelRef || typeof step.kernelRef !== 'object' || Array.isArray(step.kernelRef)) {
      throw new Error(`${prefix}.kernelRef {id, version, digest} is required for non-cast steps.`);
    }
    assertString(step.kernelRef.id, `${prefix}.kernelRef.id`);
    if (!isSemver(step.kernelRef.version)) {
      throw new Error(`${prefix}.kernelRef.version must be semver.`);
    }
    if (!isSha256Digest(step.kernelRef.digest)) {
      throw new Error(`${prefix}.kernelRef.digest must match sha256:<64-hex>.`);
    }

    try {
      buildKernelRefFromKernelEntry(kernel, entry);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`${prefix}.kernel cannot be content-pinned: ${message}`);
    }
    if (!isKernelRefBoundToKernel(step.kernelRef, kernel, entry)) {
      throw new Error(
        `${prefix}.kernelRef must match kernel binding "${kernel}#${entry}" (id/version/digest).`
      );
    }
  }
}

function applyConverterInferenceOverrides(manifestInference, converterConfig, context) {
  const overrideKernelPath = readConverterKernelPathOverride(converterConfig);
  if (overrideKernelPath) {
    manifestInference.defaultKernelPath = overrideKernelPath;
  }
  const sessionDefaults = readConverterSessionDefaultsOverride(converterConfig);
  const execution = readConverterExecutionOverride(converterConfig);
  if (sessionDefaults) {
    manifestInference.sessionDefaults = sessionDefaults;
  }
  if (execution) {
    manifestInference.execution = execution;
  }

  const layerPatternType = String(manifestInference?.layerPattern?.type ?? '').trim().toLowerCase();
  const hasCustomConvLayers = layerPatternType === 'custom'
    && Array.isArray(manifestInference?.layerPattern?.layerTypes)
    && manifestInference.layerPattern.layerTypes.some((type) => {
      const normalized = String(type ?? '').trim().toLowerCase();
      return normalized === 'conv' || normalized === 'convolution' || normalized === 'liv_conv';
    });

  if (execution && !manifestInference.sessionDefaults) {
    throw new Error(
      'converterConfig.inference.execution requires converterConfig.inference.sessionDefaults.'
    );
  }

  if (!manifestInference.execution) {
    assertNonExecutionSessionDefaults(manifestInference);
    manifestInference.schema = null;
  }
  validateDefaultKernelPath(manifestInference, context);
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
    defaultKernelPath: null,
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

  // V1 config: explicit execution graph, no preset detection
  if (isV1Config(converterConfig)) {
    return resolveConversionPlanV1(options);
  }
  const sourceQuantization = (
    options?.sourceQuantization
    ?? converterConfig?.quantization?.weights
    ?? inferSourceWeightQuantization(tensors)
  );
  const weightOverride = converterConfig?.quantization?.weights ?? null;
  // Use normalized role dtypes for kernel-path planning only.
  // Transformer preset defaults are keyed by f16/f32 families; BF16 source
  // role dtypes should not change kernel-path selection when explicit compute precision is targeted.
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

  if (options?.modelKind === 'diffusion') {
    const manifestInference = { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'diffusion' };
    applyConverterInferenceOverrides(manifestInference, converterConfig, {
      presetId: 'diffusion',
      quantizationInfo,
    });
    return {
      modelType: 'diffusion',
      presetId: 'diffusion',
      preset: null,
      sourceQuantization,
      quantizationInfo,
      manifestQuantization,
      manifestInference,
    };
  }

  const architectureHint = options?.architectureHint ?? options?.architecture ?? '';
  const presetOverride = options?.presetOverride ?? converterConfig?.presets?.model;
  let presetId = presetOverride || detectPreset(rawConfig, architectureHint);
  if (!presetOverride && isLikelyEmbeddingGemma(rawConfig, architectureHint)) {
    presetId = 'embeddinggemma';
  }
  if (!presetId) {
    throw buildUnknownFamilyError(architectureHint, rawConfig, options?.includePresetOverrideHint === true);
  }
  const preset = resolvePreset(presetId);
  const modelType = preset.modelType;
  if (!modelType) {
    throw new Error(`Preset "${presetId}" missing modelType`);
  }

  const headDim = options?.headDim ?? options?.architectureConfig?.headDim ?? preset?.architecture?.headDim ?? null;
  if (!headDim) {
    throw new Error(options?.headDimErrorMessage || 'Missing headDim in architecture');
  }

  const manifestInference = buildManifestInference(preset, rawConfig, headDim, quantizationInfo, tensorNames);
  applyConverterInferenceOverrides(manifestInference, converterConfig, { presetId, quantizationInfo });

  return {
    modelType,
    presetId,
    preset,
    sourceQuantization,
    quantizationInfo,
    manifestQuantization,
    manifestInference,
    headDim,
  };
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
