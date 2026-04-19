import path from 'node:path';

import { createConverterConfig } from '../config/schema/index.js';
import { resolveConversionPlan } from '../converter/conversion-plan.js';
import { normalizeQuantTag } from '../converter/quantization-info.js';

function toSafeString(value) {
  if (typeof value !== 'string') return '';
  const trimmed = value.trim();
  return trimmed || '';
}

function normalizeQuantizationTag(value) {
  return normalizeQuantTag(toSafeString(value));
}

function resolveArchitectureHint(architecture) {
  if (!architecture) return '';
  if (typeof architecture === 'string') return architecture;
  return (
    toSafeString(architecture.id)
    || toSafeString(architecture.name)
    || toSafeString(architecture.type)
    || ''
  );
}

function resolveHeadDim(architecture) {
  const headDim = Number(architecture?.headDim ?? architecture?.head_dim);
  return Number.isFinite(headDim) && headDim > 0 ? headDim : null;
}

function extractSourceQuantization(manifest) {
  const explicitWeights = toSafeString(manifest?.quantizationInfo?.weights);
  if (explicitWeights) return explicitWeights;
  const explicitQuant = toSafeString(manifest?.quantization);
  if (explicitQuant) return explicitQuant;
  return normalizeQuantTag(null);
}

function buildRefreshRawConfig(manifest) {
  const baseConfig = (manifest?.config && typeof manifest.config === 'object')
    ? { ...manifest.config }
    : {};
  const manifestLayerTypes = manifest?.inference?.layerPattern?.layerTypes;
  if (Array.isArray(manifestLayerTypes) && manifestLayerTypes.length > 0) {
    return {
      ...baseConfig,
      layer_types: [...manifestLayerTypes],
    };
  }
  return baseConfig;
}

function extractTensorEntriesFromManifest(manifest) {
  if (!(manifest?.tensors && typeof manifest.tensors === 'object' && !Array.isArray(manifest.tensors))) {
    return [];
  }
  return Object.entries(manifest.tensors).map(([name, tensor]) => ({
    name,
    dtype: tensor?.dtype ?? null,
    shape: tensor?.shape ?? null,
    role: tensor?.role ?? null,
    layout: tensor?.layout ?? null,
  }));
}

export function resolveMaterializedManifestFromConversionConfig(conversionConfigInput, manifest) {
  const converterConfig = createConverterConfig(conversionConfigInput);
  const tensorEntries = extractTensorEntriesFromManifest(manifest);
  const architecture = manifest?.architecture && typeof manifest.architecture === 'object'
    ? manifest.architecture
    : null;
  const plan = resolveConversionPlan({
    rawConfig: buildRefreshRawConfig(manifest),
    tensors: tensorEntries,
    converterConfig,
    sourceQuantization: normalizeQuantizationTag(extractSourceQuantization(manifest)),
    modelKind: manifest?.modelType === 'diffusion' ? 'diffusion' : 'transformer',
    architectureHint: resolveArchitectureHint(manifest?.architecture),
    architectureConfig: architecture,
    headDim: resolveHeadDim(architecture),
  });
  return {
    modelId: manifest?.modelId ?? converterConfig?.output?.modelBaseId ?? 'unknown',
    modelType: manifest?.modelType ?? plan?.modelType ?? 'transformer',
    architecture: manifest?.architecture ?? null,
    inference: plan?.manifestInference ?? null,
  };
}

export function inferConversionConfigModelId(configPath, conversionConfigInput) {
  const configuredId = toSafeString(conversionConfigInput?.output?.modelBaseId);
  if (configuredId) return configuredId;
  return path.basename(String(configPath), path.extname(String(configPath)));
}
