import { resolveKernelPath } from '../config/kernel-path-loader.js';
import { detectPreset, resolvePreset } from '../config/loader.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../config/schema/index.js';
import { buildManifestInference } from './manifest-inference.js';
import {
  buildQuantizationInfo,
  resolveManifestQuantization,
  resolveModelId,
} from './quantization-info.js';
import { sanitizeModelId } from './core.js';
import { classifyTensorRole } from '../storage/rdrr-format.js';

const SUPPORTED_MODEL_FAMILIES = 'gemma2, gemma3, embeddinggemma, modernbert, llama3, qwen3, mixtral, deepseek, mamba, gpt-oss';

function normalizeWeightDtype(dtype) {
  const upper = String(dtype || '').toUpperCase();
  return upper === 'BF16' ? 'F16' : upper;
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
    `Supported model families: ${SUPPORTED_MODEL_FAMILIES}`
  );
}

export function inferSourceWeightQuantization(tensors) {
  if (!Array.isArray(tensors) || tensors.length === 0) {
    return 'f16';
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
  if (dtypes.size === 0) return 'f16';
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
  try {
    resolveKernelPath(inference.defaultKernelPath);
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
}

export function resolveConversionPlan(options) {
  const rawConfig = options?.rawConfig || {};
  const tensors = Array.isArray(options?.tensors) ? options.tensors : [];
  const tensorNames = options?.tensorNames ?? tensors.map((tensor) => tensor.name);
  const converterConfig = options?.converterConfig;
  const sourceQuantization = (
    options?.sourceQuantization
    ?? converterConfig?.quantization?.weights
    ?? inferSourceWeightQuantization(tensors)
  );
  const weightOverride = converterConfig?.quantization?.weights ?? null;
  const embedDtypeRaw = findTensorDtypeByRole(tensors, 'embedding');
  const lmHeadDtypeRaw = findTensorDtypeByRole(tensors, 'lm_head');
  const hasVision = hasAnyTensorPattern(tensors, ['vision_', 'vision_tower', 'vision_model', 'image_encoder']);
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
    return {
      modelType: 'diffusion',
      presetId: 'diffusion',
      preset: null,
      sourceQuantization,
      quantizationInfo,
      manifestQuantization,
      manifestInference: { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'diffusion' },
    };
  }

  const architectureHint = options?.architectureHint ?? options?.architecture ?? '';
  const presetOverride = options?.presetOverride ?? converterConfig?.presets?.model;
  const presetId = presetOverride || detectPreset(rawConfig, architectureHint);
  if (presetId === 'transformer') {
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
  validateDefaultKernelPath(manifestInference, { presetId, quantizationInfo });

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

  const baseModelId = explicitModelId ?? converterConfig?.output?.modelId ?? detectedModelId ?? fallbackModelId;
  if (!baseModelId) return null;
  const resolved = sanitizeOnly
    ? baseModelId
    : resolveModelId(baseModelId, detectedModelId ?? baseModelId, quantizationInfo?.variantTag);
  return sanitizeModelId(resolved);
}
