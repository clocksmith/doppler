import { ggufBlockBytes, ggufBlockFormat, isGgufBlockDtype, isGgufBlockSourceDtype } from '../formats/gguf/block-storage.js';
import { getDevice } from '../gpu/device.js';
import { DTYPE_SIZES } from '../config/schema/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { getLargeWeightMaxBytes } from './manifest-config.js';

export const MAX_SPLIT4_EMBEDDING_SECTIONS = 4;
export const MAX_SPLIT8_EMBEDDING_SECTIONS = 8;
export const MAX_SPLIT_EMBEDDING_SECTIONS = MAX_SPLIT8_EMBEDDING_SECTIONS;
export const SPLIT_EMBEDDING_STORAGE_BUFFER_OVERHEAD = 2;

export function getEmbeddingFloatDtype(location, embeddingKernel = null) {
  // Native packed embeddings keep their source encoding; the historical helper name also covers these.
  if (isGgufBlockSourceDtype(location?.dtype)) return ggufBlockFormat(location.dtype).variant;
  if (
    String(location?.dtype ?? '').toUpperCase() === 'BF16'
    && embeddingKernel?.kernel === 'gather_bf16.wgsl'
    && embeddingKernel?.entry === 'main'
  ) {
    return 'bf16';
  }
  return selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location?.dtype,
  });
}

// Materialization follows the selected execution graph, never an uncompiled patch.
export function resolveEmbeddingKernel(manifest, resolvedKernelPaths = []) {
  const declared = manifest?.inference?.execution?.kernels?.embed ?? null;
  if (!Array.isArray(resolvedKernelPaths)) throw new Error('[Loader] Resolved kernel paths must be an array.');
  if (resolvedKernelPaths.length === 0) return declared;
  // Legacy matmul-only paths do not own embedding selection.
  if (resolvedKernelPaths.every(path => path?.preLayer == null)) return declared;
  const selected = resolvedKernelPaths.map(path => {
    const steps = path?.preLayer?.filter(step => step.op === 'embed') ?? [];
    if (steps.length !== 1) throw new Error('[Loader] Each resolved path requires exactly one embedding step.');
    const step = steps[0];
    if (typeof step.kernel !== 'string' || typeof step.entry !== 'string') {
      throw new Error('[Loader] Resolved embedding kernel and entry are required.');
    }
    return { kernel: step.kernel, entry: step.entry };
  });
  if (selected.some(step => step.kernel !== selected[0].kernel || step.entry !== selected[0].entry)) {
    throw new Error('[Loader] Reachable execution paths disagree on embedding materialization.');
  }
  return selected[0];
}

export function expectsPackedF16SplitGpuEmbeddingKernel(kernel) {
  return kernel?.kernel === 'gather_split4_packed_f16_vec4_f32_out.wgsl'
    && kernel?.entry === 'gather_vec4_f32_out';
}

export function expectsSplitGpuEmbeddingKernel(embeddingKernel) {
  return getSplitGpuEmbeddingKernelSectionCount(embeddingKernel) > 0;
}

export function getSplitGpuEmbeddingKernelSectionCount(embeddingKernel) {
  if (expectsPackedF16SplitGpuEmbeddingKernel(embeddingKernel)) return MAX_SPLIT4_EMBEDDING_SECTIONS;
  const kernel = embeddingKernel?.kernel;
  const entry = embeddingKernel?.entry;
  if (kernel === 'gather_split4_f16_vec4_f32_out.wgsl' && entry === 'gather_vec4_f32_out') {
    return MAX_SPLIT4_EMBEDDING_SECTIONS;
  }
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

export function canAutoSplitGpuEmbedding({
  location,
  embeddingKernel = null,
  materializedDtype = null,
  hostHasShaderF16 = true,
  preserveF32Embeddings = false,
} = {}) {
  if (preserveF32Embeddings || hostHasShaderF16 === false) {
    return false;
  }
  const kernel = embeddingKernel?.kernel ?? null;
  if (kernel !== null && kernel !== 'gather.wgsl' && kernel !== 'gather_f16.wgsl' && kernel !== 'gather_f16_vec4.wgsl') {
    return false;
  }
  const dtype = materializedDtype ?? getEmbeddingFloatDtype(location, embeddingKernel);
  if (dtype !== 'f16') {
    return false;
  }
  if (!Array.isArray(location?.shape) || location.shape.length !== 2) {
    return false;
  }
  const [, hidden] = location.shape;
  if (!Number.isSafeInteger(hidden) || hidden <= 0 || hidden % 4 !== 0) {
    return false;
  }
  if ((location?.layout ?? 'row') !== 'row' || Boolean(location?.sourceTransform)) {
    return false;
  }
  return true;
}

export function resolveAutoSplitSectionCap(requiredSplitSections, device = getDevice()) {
  if (!Number.isFinite(requiredSplitSections) || requiredSplitSections <= 0) {
    return 0;
  }
  const maxStorageBuffers = device?.limits?.maxStorageBuffersPerShaderStage;
  if (
    requiredSplitSections <= MAX_SPLIT4_EMBEDDING_SECTIONS
    && (!Number.isFinite(maxStorageBuffers)
      || maxStorageBuffers >= getSplitGpuEmbeddingRequiredStorageBuffers(MAX_SPLIT4_EMBEDDING_SECTIONS))
  ) {
    return MAX_SPLIT4_EMBEDDING_SECTIONS;
  }
  if (
    requiredSplitSections <= MAX_SPLIT8_EMBEDDING_SECTIONS
    && (!Number.isFinite(maxStorageBuffers)
      || maxStorageBuffers >= getSplitGpuEmbeddingRequiredStorageBuffers(MAX_SPLIT8_EMBEDDING_SECTIONS))
  ) {
    return MAX_SPLIT8_EMBEDDING_SECTIONS;
  }
  return 0;
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

function estimateEmbeddingTensorBytes(location, embeddingKernel = null, materializedDtype = null) {
  if (!location?.shape || location.shape.length !== 2) {
    return null;
  }
  const [rows, hidden] = location.shape;
  if (!Number.isFinite(rows) || rows <= 0 || !Number.isFinite(hidden) || hidden <= 0) {
    return null;
  }
  const dtype = materializedDtype ?? getEmbeddingFloatDtype(location, embeddingKernel);
  const packed = isGgufBlockDtype(dtype);
  const tensorSizeBytes = packed ? ggufBlockBytes(location.shape, dtype) : rows * hidden * DTYPE_SIZES[dtype];
  const bytesPerElement = packed ? tensorSizeBytes / (rows * hidden) : DTYPE_SIZES[dtype];
  if (!Number.isFinite(bytesPerElement) || bytesPerElement <= 0) {
    return null;
  }
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

export function createGpuResidentEmbeddingLimitError({ name, location, embeddingKernel = null, materializedDtype = null }) {
  const estimate = estimateEmbeddingTensorBytes(location, embeddingKernel, materializedDtype);
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
  const explicitSplitKernel = expectsSplitGpuEmbeddingKernel(embeddingKernel);
  const autoSplitEligible = !explicitSplitKernel && canAutoSplitGpuEmbedding({
    location,
    embeddingKernel,
    materializedDtype,
  });
  const splitKernelExpected = explicitSplitKernel || autoSplitEligible;
  const activeSplitKernelMaxSections = getSplitGpuEmbeddingKernelSectionCount(embeddingKernel) || null;
  const maxSplitEmbeddingSections = getMaxSplitGpuEmbeddingSectionsForDevice(embeddingKernel, device);
  const effectiveMaxSplitSections = explicitSplitKernel
    ? maxSplitEmbeddingSections
    : resolveAutoSplitSectionCap(requiredSplitSections, device);
  if (
    splitKernelExpected
    && Number.isFinite(requiredSplitSections)
    && requiredSplitSections <= effectiveMaxSplitSections
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
  const embeddingKernel = resolveEmbeddingKernel(manifest, options.resolvedKernelPaths ?? []);
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
