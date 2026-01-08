/**
 * Pipeline Cache - GPU pipeline creation and caching
 *
 * Handles creation and caching of compute pipelines, bind group layouts,
 * and pipeline layouts for kernel execution.
 *
 * @module gpu/kernels/pipeline-cache
 */

import { getDevice, getKernelCapabilities } from '../device.js';
import { getKernelConfig } from './kernel-configs.js';
import { getShaderModule } from './shader-cache.js';
import { hasRequiredFeatures } from './feature-check.js';
import { trace } from '../../debug/index.js';

// ============================================================================
// Caches
// ============================================================================

/** @type {Map<string, GPUComputePipeline>} Compiled pipeline cache */
const pipelineCache = new Map();

/** @type {Map<string, GPUBindGroupLayout>} Bind group layout cache */
const bindGroupLayoutCache = new Map();

/** @type {Map<string, GPUPipelineLayout>} Pipeline layout cache */
const pipelineLayoutCache = new Map();

// ============================================================================
// Bind Group Layout
// ============================================================================

/**
 * Get or create a cached bind group layout.
 * @param {string} label
 * @param {GPUBindGroupLayoutEntry[]} entries
 * @param {GPUDevice | null} [deviceOverride]
 * @returns {GPUBindGroupLayout}
 */
export function getOrCreateBindGroupLayout(
  label,
  entries,
  deviceOverride = null
) {
  const cached = bindGroupLayoutCache.get(label);
  if (cached) {
    return cached;
  }

  const device = deviceOverride || getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }

  const layout = device.createBindGroupLayout({ label, entries });
  bindGroupLayoutCache.set(label, layout);
  return layout;
}

// ============================================================================
// Pipeline Layout
// ============================================================================

/**
 * Get or create a cached pipeline layout.
 * @param {string} label
 * @param {GPUBindGroupLayout[]} bindGroupLayouts
 * @param {GPUDevice | null} [deviceOverride]
 * @returns {GPUPipelineLayout}
 */
export function getOrCreatePipelineLayout(
  label,
  bindGroupLayouts,
  deviceOverride = null
) {
  const cached = pipelineLayoutCache.get(label);
  if (cached) {
    return cached;
  }

  const device = deviceOverride || getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }

  const layout = device.createPipelineLayout({
    label,
    bindGroupLayouts,
  });

  pipelineLayoutCache.set(label, layout);
  return layout;
}

// ============================================================================
// Pipeline Creation
// ============================================================================

/**
 * Synchronously get a cached pipeline, or null if not cached.
 * Use this for fast path when you know the pipeline should be warm.
 * @param {string} operation
 * @param {string} variant
 * @returns {GPUComputePipeline | null}
 */
export function getCachedPipeline(
  operation,
  variant
) {
  const cacheKey = `${operation}:${variant}`;
  return pipelineCache.get(cacheKey) || null;
}

/**
 * Get a pipeline, using synchronous cache lookup when available.
 * Falls back to async compilation if not cached.
 * This is the preferred way to get pipelines in hot paths.
 * @param {string} operation
 * @param {string} variant
 * @param {GPUBindGroupLayout | null} [bindGroupLayout]
 * @returns {Promise<GPUComputePipeline>}
 */
export async function getPipelineFast(
  operation,
  variant,
  bindGroupLayout = null
) {
  const cached = getCachedPipeline(operation, variant);
  if (cached) {
    return cached;
  }
  return createPipeline(operation, variant, bindGroupLayout);
}

/**
 * Create a compute pipeline for a kernel
 * @param {string} operation
 * @param {string} variant
 * @param {GPUBindGroupLayout | null} [bindGroupLayout]
 * @returns {Promise<GPUComputePipeline>}
 */
export async function createPipeline(
  operation,
  variant,
  bindGroupLayout = null
) {
  const cacheKey = `${operation}:${variant}`;

  // Return cached pipeline if available
  if (pipelineCache.has(cacheKey)) {
    return pipelineCache.get(cacheKey);
  }

  const device = getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }

  const config = getKernelConfig(operation, variant);
  const capabilities = getKernelCapabilities();

  // Verify requirements
  if (!hasRequiredFeatures(config.requires, capabilities)) {
    throw new Error(
      `Kernel ${operation}/${variant} requires features: ${config.requires.join(', ')}`
    );
  }

  trace.kernels(
    `KernelLayout: ${operation}/${variant} file=${config.shaderFile} entry=${config.entryPoint} ` +
      `workgroup=[${config.workgroupSize.join(',')}] requires=` +
      `${config.requires.length > 0 ? config.requires.join('|') : 'none'}`
  );

  // Compile or reuse shader module
  const shaderModule = await getShaderModule(device, config.shaderFile, `${operation}_${variant}`);

  // Create pipeline
  const layoutLabel = bindGroupLayout?.label || `${operation}_${variant}_layout`;
  /** @type {GPUComputePipelineDescriptor} */
  const pipelineDescriptor = {
    label: `${operation}_${variant}_pipeline`,
    layout: bindGroupLayout
      ? getOrCreatePipelineLayout(layoutLabel, [bindGroupLayout], device)
      : 'auto',
    compute: {
      module: shaderModule,
      entryPoint: config.entryPoint,
    },
  };

  const pipeline = await device.createComputePipelineAsync(pipelineDescriptor);
  pipelineCache.set(cacheKey, pipeline);

  return pipeline;
}

// ============================================================================
// Cache Management
// ============================================================================

/**
 * Clear the pipeline caches
 * @returns {void}
 */
export function clearPipelineCaches() {
  pipelineCache.clear();
  bindGroupLayoutCache.clear();
  pipelineLayoutCache.clear();
}

/**
 * Get pipeline cache statistics
 * @returns {{ pipelines: number, bindGroupLayouts: number, pipelineLayouts: number }}
 */
export function getPipelineCacheStats() {
  return {
    pipelines: pipelineCache.size,
    bindGroupLayouts: bindGroupLayoutCache.size,
    pipelineLayouts: pipelineLayoutCache.size,
  };
}
