import { DEFAULT_RUNTIME_CONFIG } from '../../config/schema/index.js';
import { compileExecutionV1 } from '../pipelines/text/execution-v1.js';

export function hasF16SubgroupLaneSupport(capabilities) {
  return capabilities?.hasF16 === true && capabilities?.hasSubgroups === true;
}

function buildPlatform(capabilities) {
  return {
    id: 'model-lane-runtime',
    vendor: capabilities?.adapterInfo?.vendor ?? 'unknown',
    architecture: capabilities?.adapterInfo?.architecture ?? 'unknown',
  };
}

function getManifest(manifestByModelId, modelId) {
  if (!manifestByModelId || !modelId) return null;
  if (typeof manifestByModelId.get === 'function') return manifestByModelId.get(modelId) ?? null;
  return manifestByModelId[modelId] ?? null;
}

function cloneManifest(manifest) {
  return typeof structuredClone === 'function'
    ? structuredClone(manifest)
    : JSON.parse(JSON.stringify(manifest));
}

export function assertExecutionLaneManifestSupported(entry, manifest, capabilities, options = {}) {
  const normalizedManifest = typeof options.normalizeManifest === 'function'
    ? options.normalizeManifest(cloneManifest(manifest))
    : cloneManifest(manifest);
  compileExecutionV1({
    manifestInference: normalizedManifest.inference,
    modelId: normalizedManifest.modelId ?? entry?.modelId ?? 'unknown',
    numLayers: normalizedManifest.architecture?.numLayers ?? 0,
    headDim: normalizedManifest.architecture?.headDim ?? null,
    capabilities,
    platform: options.platform ?? buildPlatform(capabilities),
    kernelPathPolicy: options.kernelPathPolicy ?? DEFAULT_RUNTIME_CONFIG.inference.kernelPathPolicy,
  });
}

function unsupportedError(entry, fallback, preferredError, fallbackError) {
  return new Error(
    `${entry?.modelId ?? 'unknown'}: preferred lane is unsupported on this device (${preferredError.message}); ` +
    `${fallback?.modelId ?? 'unknown'}: fallback lane is also unsupported (${fallbackError.message}).`
  );
}

export function selectExecutionLaneForCapabilities(entry, options = {}) {
  const capabilities = options.capabilities;
  const manifestByModelId = options.manifestByModelId ?? options.manifests;
  const fallback = entry?.demoFallbackVariant ?? null;
  const manifest = getManifest(manifestByModelId, entry?.modelId);
  if (!manifest) {
    throw new Error(`${entry?.modelId ?? 'unknown'}: execution lane preflight missing manifest.`);
  }

  const assertSupported = (candidate, candidateManifest) => assertExecutionLaneManifestSupported(
    candidate,
    candidateManifest,
    capabilities,
    {
      normalizeManifest: options.normalizeManifest,
      kernelPathPolicy: options.kernelPathPolicy,
      platform: options.platform,
    }
  );

  if (!fallback) {
    assertSupported(entry, manifest);
    return {
      entry,
      selectedModelId: entry?.modelId ?? null,
      usedFallback: false,
      rejected: [],
      reason: 'primary lane supported',
    };
  }

  const fallbackManifest = getManifest(manifestByModelId, fallback.modelId);
  if (!fallbackManifest) {
    throw new Error(`${fallback.modelId ?? 'unknown'}: execution lane fallback preflight missing manifest.`);
  }

  if (!hasF16SubgroupLaneSupport(capabilities)) {
    assertSupported(fallback, fallbackManifest);
    return {
      entry: fallback,
      selectedModelId: fallback.modelId ?? null,
      usedFallback: true,
      rejected: [{
        modelId: entry?.modelId ?? null,
        reason: 'device lacks shader-f16 plus subgroups',
      }],
      reason: 'fallback selected because preferred lane requires shader-f16 plus subgroups',
    };
  }

  try {
    assertSupported(entry, manifest);
    return {
      entry,
      selectedModelId: entry?.modelId ?? null,
      usedFallback: false,
      rejected: [],
      reason: 'preferred lane supported',
    };
  } catch (preferredError) {
    try {
      assertSupported(fallback, fallbackManifest);
      return {
        entry: fallback,
        selectedModelId: fallback.modelId ?? null,
        usedFallback: true,
        rejected: [{
          modelId: entry?.modelId ?? null,
          reason: preferredError.message,
        }],
        reason: 'fallback selected after preferred lane preflight rejection',
      };
    } catch (fallbackError) {
      throw unsupportedError(entry, fallback, preferredError, fallbackError);
    }
  }
}
