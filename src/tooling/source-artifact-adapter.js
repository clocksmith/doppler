import {
  inferSourceWeightQuantization,
  resolveConvertedModelId,
} from '../converter/conversion-plan.js';
import {
  normalizeQuantTag,
  resolveEffectiveQuantizationInfo,
  resolveManifestQuantization,
} from '../converter/quantization-info.js';
import { resolveTensorRole } from '../formats/rdrr/index.js';
import { log } from '../debug/index.js';
import {
  buildSourceRuntimeBundle,
} from './source-runtime-bundle.js';
import {
  createSourceRuntimeManifestConfig,
  createSourceRuntimeManifestInference,
} from './source-runtime-converter-config.js';

export const SOURCE_ARTIFACT_KIND_SAFETENSORS = 'safetensors';
export const SOURCE_ARTIFACT_KIND_GGUF = 'gguf';
export const SOURCE_ARTIFACT_KIND_TFLITE = 'tflite';
export const SOURCE_ARTIFACT_KIND_LITERT_TASK = 'litert-task';
export const SOURCE_ARTIFACT_KIND_LITERTLM = 'litertlm';

const SUPPORTED_SOURCE_DTYPES = new Set([
  'F32',
  'F16',
  'BF16',
  'Q4_K',
  'Q4_K_M',
  'Q6_K',
]);

const SOURCE_QUANT_COMPUTE_MAP = {
  F16: 'f16',
  BF16: 'f32',
  F32: 'f32',
  Q4_K: 'f32',
  Q4_K_M: 'f32',
  Q6_K: 'f32',
};
const SOURCE_COMPUTE_DEFAULT = 'f16';

function normalizeText(value) {
  return String(value || '').trim();
}

function normalizePath(value) {
  return normalizeText(value).replace(/\\/g, '/').replace(/\/+$/, '');
}

function resolvePathBasename(value, fallback) {
  const normalized = normalizePath(value);
  if (!normalized) {
    return fallback;
  }
  const segments = normalized.split('/').filter(Boolean);
  const last = segments.length > 0 ? segments[segments.length - 1] : normalized;
  const dotIndex = last.lastIndexOf('.');
  const basename = dotIndex > 0 ? last.slice(0, dotIndex) : last;
  return basename || fallback;
}

function normalizeStoredQuantization(value) {
  const normalized = normalizeText(value).toUpperCase();
  if (!normalized) {
    return null;
  }
  return normalizeQuantTag(normalized);
}

function resolveDirectSourceModelType(parsedArtifact, fallbackModelKind) {
  const rawConfig = parsedArtifact?.config;
  const explicitModelType = normalizeText(parsedArtifact?.modelType);
  if (explicitModelType) {
    return explicitModelType;
  }
  const configModelType = normalizeText(
    rawConfig?.model_type
    ?? rawConfig?.text_config?.model_type
  );
  if (configModelType) {
    return configModelType;
  }
  const fallback = normalizeText(fallbackModelKind);
  return fallback || 'transformer';
}

function inferRoleQuantization(tensors, targetRole) {
  for (const tensor of Array.isArray(tensors) ? tensors : []) {
    if (resolveTensorRole(tensor) !== targetRole) {
      continue;
    }
    const normalized = normalizeStoredQuantization(tensor?.dtype);
    if (normalized) {
      return normalized;
    }
  }
  return null;
}

export function resolveDirectSourceRuntimePlan(options = {}) {
  const parsedArtifact = options.parsedArtifact;
  if (!parsedArtifact || typeof parsedArtifact !== 'object' || Array.isArray(parsedArtifact)) {
    throw new Error('direct-source runtime: parsedArtifact must be an object.');
  }

  const sourceQuantization = normalizeText(options.sourceQuantization)
    || inferSourceQuantizationForSourceRuntime(
      parsedArtifact.tensors,
      parsedArtifact.sourceKind || 'direct-source',
      { logCategory: options.logCategory }
    );
  const computePrecision = resolveSourceRuntimeComputePrecision(
    parsedArtifact.tensors,
    sourceQuantization
  );
  const visionQuantization = inferRoleQuantization(parsedArtifact.tensors, 'vision');
  const audioQuantization = inferRoleQuantization(parsedArtifact.tensors, 'audio');
  const projectorQuantization = inferRoleQuantization(parsedArtifact.tensors, 'projector');
  const baseQuantizationInfo = {
    compute: computePrecision,
    ...(visionQuantization ? { vision: visionQuantization } : {}),
    ...(audioQuantization ? { audio: audioQuantization } : {}),
    ...(projectorQuantization ? { projector: projectorQuantization } : {}),
  };
  const quantizationInfo = resolveEffectiveQuantizationInfo(
    baseQuantizationInfo,
    parsedArtifact.tensors
  );
  const manifestQuantization = resolveManifestQuantization(
    quantizationInfo?.weights,
    normalizeText(sourceQuantization).toUpperCase() || 'F16'
  );
  const baseManifestConfig = createSourceRuntimeManifestConfig(parsedArtifact.config ?? null);
  const explicitManifestConfig = (
    parsedArtifact?.manifestConfig
    && typeof parsedArtifact.manifestConfig === 'object'
    && !Array.isArray(parsedArtifact.manifestConfig)
  )
    ? parsedArtifact.manifestConfig
    : null;
  const explicitManifestInference = (
    parsedArtifact?.manifestInference
    && typeof parsedArtifact.manifestInference === 'object'
    && !Array.isArray(parsedArtifact.manifestInference)
  )
    ? parsedArtifact.manifestInference
    : null;

  return {
    modelType: resolveDirectSourceModelType(parsedArtifact, options.modelKind),
    manifestConfig: explicitManifestConfig
      ? {
        ...baseManifestConfig,
        ...explicitManifestConfig,
      }
      : baseManifestConfig,
    manifestInference: explicitManifestInference ?? createSourceRuntimeManifestInference(parsedArtifact.config ?? null),
    sourceQuantization,
    quantizationInfo,
    manifestQuantization,
    executionVersion: 'v1',
  };
}

export function normalizeSourceArtifactKind(value) {
  const normalized = normalizeText(value).toLowerCase();
  return normalized || null;
}

export function assertDirectSourceRuntimeSupportedKind(sourceKind, label = 'direct-source runtime') {
  const normalized = normalizeSourceArtifactKind(sourceKind);
  if (
    normalized === SOURCE_ARTIFACT_KIND_SAFETENSORS
    || normalized === SOURCE_ARTIFACT_KIND_GGUF
    || normalized === SOURCE_ARTIFACT_KIND_TFLITE
    || normalized === SOURCE_ARTIFACT_KIND_LITERT_TASK
    || normalized === SOURCE_ARTIFACT_KIND_LITERTLM
  ) {
    return normalized;
  }
  if (!normalized) {
    throw new Error(`${label}: sourceKind is required.`);
  }
  throw new Error(`${label}: unsupported direct-source artifact kind "${sourceKind}". Convert to RDRR first.`);
}

export function assertSupportedSourceDtypes(tensors, sourceKind) {
  const unsupported = new Set();
  for (const tensor of Array.isArray(tensors) ? tensors : []) {
    const dtype = normalizeText(tensor?.dtype).toUpperCase();
    if (!dtype) {
      unsupported.add('(empty)');
      continue;
    }
    if (!SUPPORTED_SOURCE_DTYPES.has(dtype)) {
      unsupported.add(dtype);
    }
  }
  if (unsupported.size > 0) {
    throw new Error(
      `Unsupported ${sourceKind} tensor dtypes for direct-source runtime: ` +
      `${Array.from(unsupported).sort((left, right) => left.localeCompare(right)).join(', ')}. ` +
      'Convert to RDRR first for this model.'
    );
  }
}

export function inferSourceQuantizationForSourceRuntime(tensors, sourceKind, options = {}) {
  try {
    return inferSourceWeightQuantization(tensors);
  } catch (error) {
    const dtypes = new Set();
    for (const tensor of Array.isArray(tensors) ? tensors : []) {
      const dtype = normalizeText(tensor?.dtype).toUpperCase();
      if (dtype) {
        dtypes.add(dtype);
      }
    }
    const hasLowPrecision = dtypes.has('F16') || dtypes.has('BF16');
    const onlyLowAndF32 = dtypes.size > 0 && Array.from(dtypes).every(
      (dtype) => dtype === 'F16' || dtype === 'BF16' || dtype === 'F32'
    );
    if (hasLowPrecision && onlyLowAndF32) {
      const logCategory = normalizeText(options.logCategory) || 'SourceArtifactAdapter';
      log.warn(
        logCategory,
        `Mixed ${sourceKind} tensor dtypes detected (${Array.from(dtypes).sort((left, right) => left.localeCompare(right)).join(', ')}). ` +
        'Using F32 source quantization for direct-source parity.'
      );
      return 'F32';
    }
    throw error;
  }
}

export function resolveSourceRuntimeComputePrecision(tensors, sourceQuantization) {
  const dtypes = new Set();
  for (const tensor of Array.isArray(tensors) ? tensors : []) {
    const dtype = normalizeText(tensor?.dtype).toUpperCase();
    if (dtype) {
      dtypes.add(dtype);
    }
  }
  for (const dtype of dtypes) {
    if (SOURCE_QUANT_COMPUTE_MAP[dtype] === 'f32') {
      return 'f32';
    }
  }
  const normalizedSourceQuantization = normalizeText(sourceQuantization).toUpperCase();
  return SOURCE_QUANT_COMPUTE_MAP[normalizedSourceQuantization] ?? SOURCE_COMPUTE_DEFAULT;
}

export function resolveSourceRuntimeModelIdHint(options = {}) {
  const requestedModelId = normalizeText(options.requestedModelId);
  const sourceKind = assertDirectSourceRuntimeSupportedKind(
    options.sourceKind,
    options.label || 'direct-source runtime'
  );
  const plan = options.plan;
  if (!plan || typeof plan !== 'object') {
    throw new Error('direct-source runtime: plan is required to resolve modelId.');
  }
  if (requestedModelId) {
    return resolveConvertedModelId({
      explicitModelId: requestedModelId,
      converterConfig: null,
      detectedModelId: requestedModelId,
      quantizationInfo: plan.quantizationInfo,
    }) || requestedModelId;
  }
  const detectedModelId = resolvePathBasename(
    options.sourcePath,
    `${sourceKind}-runtime`
  );
  return resolveConvertedModelId({
    explicitModelId: detectedModelId,
    converterConfig: null,
    detectedModelId,
    quantizationInfo: plan.quantizationInfo,
  }) || detectedModelId;
}

export async function resolveSourceRuntimeBundleFromParsedArtifact(options = {}) {
  const parsedArtifact = options.parsedArtifact;
  const runtimeLabel = normalizeText(options.runtimeLabel) || 'direct-source runtime';
  if (!parsedArtifact || typeof parsedArtifact !== 'object' || Array.isArray(parsedArtifact)) {
    throw new Error(`${runtimeLabel}: parsedArtifact must be an object.`);
  }
  const sourceKind = assertDirectSourceRuntimeSupportedKind(parsedArtifact.sourceKind, runtimeLabel);
  const hashFileEntries = options.hashFileEntries;
  if (typeof hashFileEntries !== 'function') {
    throw new Error(`${runtimeLabel}: hashFileEntries(entries, hashAlgorithm) is required.`);
  }
  if (options.quantization != null) {
    throw new Error(
      `${runtimeLabel}: converter-style quantization overrides are not supported for direct-source artifacts. ` +
      'Use the source artifact as-is, or run convert to emit a new RDRR artifact.'
    );
  }

  const sourceQuantization = normalizeText(parsedArtifact.sourceQuantization)
    || inferSourceQuantizationForSourceRuntime(parsedArtifact.tensors, sourceKind, {
      logCategory: options.logCategory,
    });

  assertSupportedSourceDtypes(parsedArtifact.tensors, sourceKind);

  const plan = resolveDirectSourceRuntimePlan({
    parsedArtifact,
    sourceQuantization,
    modelKind: normalizeText(options.modelKind) || 'transformer',
    logCategory: options.logCategory,
  });
  const modelId = resolveSourceRuntimeModelIdHint({
    requestedModelId: options.requestedModelId,
    plan,
    sourceKind,
    sourcePath: parsedArtifact.sourcePathForModelId,
    label: runtimeLabel,
  });
  const hashAlgorithm = plan.manifestConfig.hashAlgorithm;
  const sourceFiles = await hashFileEntries(parsedArtifact.sourceFiles, hashAlgorithm);
  const auxiliaryFiles = await hashFileEntries(parsedArtifact.auxiliaryFiles, hashAlgorithm);
  const { model, shardSources } = await buildSourceRuntimeBundle({
    modelId,
    modelName: modelId,
    modelType: plan.modelType,
    sourceKind,
    architecture: parsedArtifact.architecture,
    architectureHint: parsedArtifact.architectureHint,
    rawConfig: parsedArtifact.config,
    manifestConfig: plan.manifestConfig ?? null,
    inference: plan.manifestInference,
    tensors: parsedArtifact.tensors,
    embeddingPostprocessor: parsedArtifact.embeddingPostprocessor ?? null,
    sourceFiles,
    auxiliaryFiles,
    sourceQuantization,
    quantizationInfo: plan.quantizationInfo,
    manifestQuantization: plan.manifestQuantization,
    hashAlgorithm,
    tokenizerJson: parsedArtifact.tokenizerJson,
    tokenizerConfig: parsedArtifact.tokenizerConfig,
    tokenizerModelName: parsedArtifact.tokenizerModelName,
    tokenizerJsonPath: parsedArtifact.tokenizerJsonPath,
    tokenizerConfigPath: parsedArtifact.tokenizerConfigPath,
    tokenizerModelPath: parsedArtifact.tokenizerModelPath,
  });

  return {
    model,
    manifest: model,
    shardSources,
    sourceKind,
    sourceQuantization,
    sourceFiles,
    auxiliaryFiles,
    hashAlgorithm,
    modelId,
    plan,
    manifestConfig: plan.manifestConfig,
  };
}
