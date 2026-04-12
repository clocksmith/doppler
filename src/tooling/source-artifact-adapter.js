import {
  inferSourceWeightQuantization,
  resolveConversionPlan,
  resolveConvertedModelId,
} from '../converter/conversion-plan.js';
import { log } from '../debug/index.js';
import {
  buildSourceRuntimeBundle,
} from './source-runtime-bundle.js';
import { createSourceRuntimeConverterConfig } from './source-runtime-converter-config.js';

export const SOURCE_ARTIFACT_KIND_SAFETENSORS = 'safetensors';
export const SOURCE_ARTIFACT_KIND_GGUF = 'gguf';
export const SOURCE_ARTIFACT_KIND_TFLITE = 'tflite';

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

export function normalizeSourceArtifactKind(value) {
  const normalized = normalizeText(value).toLowerCase();
  return normalized || null;
}

export function assertDirectSourceRuntimeSupportedKind(sourceKind, label = 'direct-source runtime') {
  const normalized = normalizeSourceArtifactKind(sourceKind);
  if (normalized === SOURCE_ARTIFACT_KIND_SAFETENSORS || normalized === SOURCE_ARTIFACT_KIND_GGUF) {
    return normalized;
  }
  if (normalized === SOURCE_ARTIFACT_KIND_TFLITE) {
    throw new Error(`${label}: .tflite direct-source artifacts are not implemented yet. Convert to RDRR first.`);
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

  const sourceQuantization = normalizeText(parsedArtifact.sourceQuantization)
    || inferSourceQuantizationForSourceRuntime(parsedArtifact.tensors, sourceKind, {
      logCategory: options.logCategory,
    });

  assertSupportedSourceDtypes(parsedArtifact.tensors, sourceKind);

  const converterConfig = createSourceRuntimeConverterConfig({
    modelId: options.requestedModelId || null,
    rawConfig: parsedArtifact.config,
    ...(options.quantization ? { quantization: options.quantization } : {}),
  });
  const plan = resolveConversionPlan({
    rawConfig: parsedArtifact.config,
    tensors: parsedArtifact.tensors,
    converterConfig,
    sourceQuantization,
    modelKind: normalizeText(options.modelKind) || 'transformer',
    architectureHint: parsedArtifact.architectureHint,
    architectureConfig: parsedArtifact.architecture,
  });
  const modelId = resolveSourceRuntimeModelIdHint({
    requestedModelId: options.requestedModelId,
    plan,
    sourceKind,
    sourcePath: parsedArtifact.sourcePathForModelId,
    label: runtimeLabel,
  });
  const hashAlgorithm = converterConfig.manifest.hashAlgorithm;
  const sourceFiles = await hashFileEntries(parsedArtifact.sourceFiles, hashAlgorithm);
  const auxiliaryFiles = await hashFileEntries(parsedArtifact.auxiliaryFiles, hashAlgorithm);
  const { manifest, shardSources } = await buildSourceRuntimeBundle({
    modelId,
    modelName: modelId,
    modelType: plan.modelType,
    sourceKind,
    architecture: parsedArtifact.architecture,
    architectureHint: parsedArtifact.architectureHint,
    rawConfig: parsedArtifact.config,
    manifestConfig: converterConfig.manifest ?? null,
    inference: plan.manifestInference,
    tensors: parsedArtifact.tensors,
    embeddingPostprocessor: parsedArtifact.embeddingPostprocessor ?? null,
    sourceFiles,
    auxiliaryFiles,
    sourceQuantization,
    quantizationInfo: plan.quantizationInfo,
    hashAlgorithm,
    tokenizerJson: parsedArtifact.tokenizerJson,
    tokenizerConfig: parsedArtifact.tokenizerConfig,
    tokenizerModelName: parsedArtifact.tokenizerModelName,
    tokenizerJsonPath: parsedArtifact.tokenizerJsonPath,
    tokenizerConfigPath: parsedArtifact.tokenizerConfigPath,
    tokenizerModelPath: parsedArtifact.tokenizerModelPath,
  });

  return {
    manifest,
    shardSources,
    sourceKind,
    sourceQuantization,
    sourceFiles,
    auxiliaryFiles,
    hashAlgorithm,
    modelId,
    plan,
    converterConfig,
  };
}
