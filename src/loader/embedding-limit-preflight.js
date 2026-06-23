import { getDevice } from '../gpu/device.js';
import { DTYPE_SIZES } from '../config/schema/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { getLargeWeightMaxBytes } from './manifest-config.js';

export const MAX_SPLIT4_EMBEDDING_SECTIONS = 4;
export const MAX_SPLIT8_EMBEDDING_SECTIONS = 8;
export const MAX_SPLIT_EMBEDDING_SECTIONS = MAX_SPLIT8_EMBEDDING_SECTIONS;
export const SPLIT_EMBEDDING_STORAGE_BUFFER_OVERHEAD = 2;

export function getEmbeddingFloatDtype(location) {
  return selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location?.dtype,
  });
}

export function expectsSplitGpuEmbeddingKernel(embeddingKernel) {
  return getSplitGpuEmbeddingKernelSectionCount(embeddingKernel) > 0;
}

export function getSplitGpuEmbeddingKernelSectionCount(embeddingKernel) {
  const kernel = embeddingKernel?.kernel;
  const entry = embeddingKernel?.entry;
  if (kernel === 'gather_split4_f16_vec4_f16_out.wgsl' && entry === 'gather_vec4_f16_out') {
    return MAX_SPLIT4_EMBEDDING_SECTIONS;
  }
  if (kernel === 'gather_split8_f16_vec4_f16_out.wgsl' && entry === 'gather_vec4_f16_out') {
    return MAX_SPLIT8_EMBEDDING_SECTIONS;
  }
  if (kernel === 'gather_split8_f16_vec4_f32_out.wgsl' && entry === 'gather_vec4_f32_out') {
    return MAX_SPLIT8_EMBEDDING_SECTIONS;
  }
  return 0;
}

export function getSplitGpuEmbeddingRequiredStorageBuffers(sectionCount) {
  return sectionCount + SPLIT_EMBEDDING_STORAGE_BUFFER_OVERHEAD;
}

export function getMaxSplitGpuEmbeddingSectionsForDevice(embeddingKernel = null, device = getDevice()) {
  const kernelSections = getSplitGpuEmbeddingKernelSectionCount(embeddingKernel);
  const maxStorageBuffersPerShaderStage = device?.limits?.maxStorageBuffersPerShaderStage;
  if (kernelSections > 0) {
    const requiredStorageBuffers = getSplitGpuEmbeddingRequiredStorageBuffers(kernelSections);
    if (
      Number.isFinite(maxStorageBuffersPerShaderStage)
      && maxStorageBuffersPerShaderStage < requiredStorageBuffers
    ) {
      return 0;
    }
    return kernelSections;
  }
  if (!Number.isFinite(maxStorageBuffersPerShaderStage)) {
    return MAX_SPLIT_EMBEDDING_SECTIONS;
  }
  return Math.max(
    0,
    Math.min(
      MAX_SPLIT_EMBEDDING_SECTIONS,
      maxStorageBuffersPerShaderStage - SPLIT_EMBEDDING_STORAGE_BUFFER_OVERHEAD
    )
  );
}

function alignByteLength(byteLength) {
  return Math.ceil(byteLength / 4) * 4;
}

function estimateEmbeddingTensorBytes(location) {
  if (!location?.shape || location.shape.length !== 2) {
    return null;
  }
  const [rows, hidden] = location.shape;
  if (!Number.isFinite(rows) || rows <= 0 || !Number.isFinite(hidden) || hidden <= 0) {
    return null;
  }
  const dtype = getEmbeddingFloatDtype(location);
  const bytesPerElement = DTYPE_SIZES[dtype];
  if (!Number.isFinite(bytesPerElement) || bytesPerElement <= 0) {
    return null;
  }
  const tensorSizeBytes = rows * hidden * bytesPerElement;
  if (!Number.isFinite(tensorSizeBytes) || tensorSizeBytes <= 0) {
    return null;
  }
  return {
    dtype,
    rows,
    hidden,
    bytesPerElement,
    tensorSizeBytes,
    alignedTensorSizeBytes: alignByteLength(tensorSizeBytes),
  };
}

export function createGpuResidentEmbeddingLimitError({ name, location, embeddingKernel = null }) {
  const estimate = estimateEmbeddingTensorBytes(location);
  if (!estimate) {
    return null;
  }
  const device = getDevice();
  const maxStorageBufferBindingSize = device?.limits?.maxStorageBufferBindingSize;
  const maxBufferSize = device?.limits?.maxBufferSize;
  const maxStorageBuffersPerShaderStage = device?.limits?.maxStorageBuffersPerShaderStage;
  if (
    !Number.isFinite(maxStorageBufferBindingSize)
    || maxStorageBufferBindingSize <= 0
    || !Number.isFinite(maxBufferSize)
    || maxBufferSize <= 0
  ) {
    return null;
  }
  const maxGpuResidentBytes = Math.min(maxStorageBufferBindingSize, maxBufferSize);
  if (estimate.alignedTensorSizeBytes <= maxGpuResidentBytes) {
    return null;
  }

  const rowBytes = estimate.hidden * estimate.bytesPerElement;
  const largeWeightMaxBytes = getLargeWeightMaxBytes();
  const rowsPerSplitSection = largeWeightMaxBytes
    ? Math.floor(largeWeightMaxBytes / rowBytes)
    : null;
  const requiredSplitSections = rowsPerSplitSection && rowsPerSplitSection > 0
    ? Math.ceil(estimate.rows / rowsPerSplitSection)
    : null;
  const splitKernelExpected = expectsSplitGpuEmbeddingKernel(embeddingKernel);
  const activeSplitKernelMaxSections = getSplitGpuEmbeddingKernelSectionCount(embeddingKernel) || null;
  const maxSplitEmbeddingSections = getMaxSplitGpuEmbeddingSectionsForDevice(embeddingKernel, device);
  if (
    splitKernelExpected
    && Number.isFinite(requiredSplitSections)
    && requiredSplitSections <= maxSplitEmbeddingSections
  ) {
    return null;
  }
  const error = new Error(
    `[Loader] Embedding "${name}" cannot be GPU-resident on this device: ` +
    `requires ${estimate.alignedTensorSizeBytes} bytes but max storage binding is ` +
    `${maxGpuResidentBytes} bytes. Configure a split embedding kernel that fits this device ` +
    'or use a device with a larger storage binding limit.'
  );
  error.details = {
    weightLoadFailure: {
      tensorName: name,
      tensorRole: location.role ?? null,
      tensorDtype: location.dtype ?? null,
      tensorShape: Array.isArray(location.shape) ? [...location.shape] : null,
      tensorSizeBytes: estimate.alignedTensorSizeBytes,
      tensorLoadStage: 'gpuResidentEmbeddingLimitPreflight',
      toGPU: true,
      streamedUpload: false,
      deviceLimitFailure: {
        kind: 'gpu_resident_embedding_exceeds_device_limit',
        maxGpuResidentBytes,
        maxStorageBufferBindingSize,
        maxBufferSize,
        maxStorageBuffersPerShaderStage: Number.isFinite(maxStorageBuffersPerShaderStage)
          ? maxStorageBuffersPerShaderStage
          : null,
        largeWeightMaxBytes,
        embeddingKernel: embeddingKernel
          ? {
              kernel: embeddingKernel.kernel ?? null,
              entry: embeddingKernel.entry ?? null,
            }
          : null,
        splitKernelExpected,
        activeSplitKernelMaxSections,
        maxSplitEmbeddingSections,
        requiredSplitSections,
      },
    },
  };
  return error;
}

function resolveLargeWeightGpuResidentOverrides(manifestOverrides, runtimeConfig) {
  const runtimeOverrides = runtimeConfig?.inference?.largeWeights?.gpuResidentOverrides;
  if (Array.isArray(runtimeOverrides)) {
    return runtimeOverrides;
  }
  if (Array.isArray(manifestOverrides)) {
    return manifestOverrides;
  }
  return null;
}

export function resolveManifestGpuResidentEmbeddingLimitError(manifest, options = {}) {
  const tensorLocations = options.storageManifest?.tensors ?? manifest?.tensors;
  if (!tensorLocations || typeof tensorLocations !== 'object') {
    return null;
  }
  const overrides = resolveLargeWeightGpuResidentOverrides(
    manifest?.inference?.largeWeights?.gpuResidentOverrides,
    options.runtimeConfig
  );
  if (!Array.isArray(overrides) || overrides.length === 0) {
    return null;
  }
  const embeddingKernel = manifest?.inference?.execution?.kernels?.embed ?? null;
  for (const name of overrides) {
    const location = tensorLocations[name];
    if (location?.role !== 'embedding') {
      continue;
    }
    const error = createGpuResidentEmbeddingLimitError({
      name,
      location,
      embeddingKernel,
    });
    if (error) {
      return error;
    }
  }
  return null;
}
