/**
 * Pipeline Cache - GPU pipeline creation and caching
 *
 * Handles creation and caching of compute pipelines, bind group layouts,
 * and pipeline layouts for kernel execution.
 *
 * @module gpu/kernels/pipeline-cache
 */

import { getDevice, getKernelCapabilities } from '../device.js';
import { getKernelConfig, type KernelConfig } from './kernel-configs.js';
import { getShaderModule } from './shader-cache.js';
import { hasRequiredFeatures } from './feature-check.js';
import { trace } from '../../debug/index.js';

// ============================================================================
// Caches
// ============================================================================

/** Compiled pipeline cache */
const pipelineCache = new Map<string, GPUComputePipeline>();

/** Bind group layout cache */
const bindGroupLayoutCache = new Map<string, GPUBindGroupLayout>();

/** Pipeline layout cache */
const pipelineLayoutCache = new Map<string, GPUPipelineLayout>();

// ============================================================================
// Bind Group Layout
// ============================================================================

/**
 * Get or create a cached bind group layout.
 */
export function getOrCreateBindGroupLayout(
  label: string,
  entries: GPUBindGroupLayoutEntry[],
  deviceOverride: GPUDevice | null = null
): GPUBindGroupLayout {
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
 */
export function getOrCreatePipelineLayout(
  label: string,
  bindGroupLayouts: GPUBindGroupLayout[],
  deviceOverride: GPUDevice | null = null
): GPUPipelineLayout {
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
 */
export function getCachedPipeline(
  operation: string,
  variant: string
): GPUComputePipeline | null {
  const cacheKey = `${operation}:${variant}`;
  return pipelineCache.get(cacheKey) || null;
}

/**
 * Get a pipeline, using synchronous cache lookup when available.
 * Falls back to async compilation if not cached.
 * This is the preferred way to get pipelines in hot paths.
 */
export async function getPipelineFast(
  operation: string,
  variant: string,
  bindGroupLayout: GPUBindGroupLayout | null = null
): Promise<GPUComputePipeline> {
  const cached = getCachedPipeline(operation, variant);
  if (cached) {
    return cached;
  }
  return createPipeline(operation, variant, bindGroupLayout);
}

/**
 * Create a compute pipeline for a kernel
 */
export async function createPipeline(
  operation: string,
  variant: string,
  bindGroupLayout: GPUBindGroupLayout | null = null
): Promise<GPUComputePipeline> {
  const cacheKey = `${operation}:${variant}`;

  // Return cached pipeline if available
  if (pipelineCache.has(cacheKey)) {
    return pipelineCache.get(cacheKey)!;
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
  const pipelineDescriptor: GPUComputePipelineDescriptor = {
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
 */
export function clearPipelineCaches(): void {
  pipelineCache.clear();
  bindGroupLayoutCache.clear();
  pipelineLayoutCache.clear();
}

/**
 * Get pipeline cache statistics
 */
export function getPipelineCacheStats(): {
  pipelines: number;
  bindGroupLayouts: number;
  pipelineLayouts: number;
} {
  return {
    pipelines: pipelineCache.size,
    bindGroupLayouts: bindGroupLayoutCache.size,
    pipelineLayouts: pipelineLayoutCache.size,
  };
}
