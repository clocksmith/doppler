

import { getDevice, getKernelCapabilities } from '../device.js';
import { getKernelConfig } from './kernel-configs.js';
import { getShaderModule } from './shader-cache.js';
import { hasRequiredFeatures } from './feature-check.js';
import { trace } from '../../debug/index.js';

// ============================================================================
// Caches
// ============================================================================


const pipelineCache = new Map();


const bindGroupLayoutCache = new Map();


const pipelineLayoutCache = new Map();

// ============================================================================
// Bind Group Layout
// ============================================================================


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


function buildPipelineCacheKey(operation, variant, constants, bindGroupLayout) {
  const constantsKey = constants
    ? Object.entries(constants).sort().map(([k, v]) => `${k}=${v}`).join('|')
    : '';
  const layoutKey = bindGroupLayout ? `:${bindGroupLayout.label || 'layout'}` : '';
  return `${operation}:${variant}${constants ? ':' + constantsKey : ''}${layoutKey}`;
}

export function getCachedPipeline(
  operation,
  variant,
  constants = null
) {
  const cacheKey = buildPipelineCacheKey(operation, variant, constants, null);
  return pipelineCache.get(cacheKey) || null;
}


export async function getPipelineFast(
  operation,
  variant,
  bindGroupLayout = null,
  constants = null
) {
  if (bindGroupLayout) {
    const layoutKey = buildPipelineCacheKey(operation, variant, constants, bindGroupLayout);
    const cached = pipelineCache.get(layoutKey);
    if (cached) return cached;
    return createPipeline(operation, variant, bindGroupLayout, constants);
  }
  const cached = getCachedPipeline(operation, variant, constants);
  if (cached) return cached;
  return createPipeline(operation, variant, null, constants);
}


export async function createPipeline(
  operation,
  variant,
  bindGroupLayout = null,
  constants = null
) {
  const constantsKey = constants
    ? Object.entries(constants).sort().map(([k, v]) => `${k}=${v}`).join('|')
    : '';
  const cacheKey = buildPipelineCacheKey(operation, variant, constants, bindGroupLayout);

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
    `${config.requires.length > 0 ? config.requires.join('|') : 'none'}` +
    `${constants ? ' constants=' + constantsKey : ''}`
  );

  // Compile or reuse shader module
  const shaderModule = await getShaderModule(device, config.shaderFile, `${operation}_${variant}`);

  // Create pipeline
  const layoutLabel = bindGroupLayout?.label || `${operation}_${variant}_layout`;

  const pipelineDescriptor = {
    label: `${operation}_${variant}_pipeline${constants ? '_' + constantsKey : ''}`,
    layout: bindGroupLayout
      ? getOrCreatePipelineLayout(layoutLabel, [bindGroupLayout], device)
      : 'auto',
    compute: {
      module: shaderModule,
      entryPoint: config.entryPoint,
      constants: constants || undefined,
    },
  };

  const pipeline = await device.createComputePipelineAsync(pipelineDescriptor);
  pipelineCache.set(cacheKey, pipeline);

  return pipeline;
}

// ============================================================================
// Cache Management
// ============================================================================


export function clearPipelineCaches() {
  pipelineCache.clear();
  bindGroupLayoutCache.clear();
  pipelineLayoutCache.clear();
}


export function getPipelineCacheStats() {
  return {
    pipelines: pipelineCache.size,
    bindGroupLayouts: bindGroupLayoutCache.size,
    pipelineLayouts: pipelineLayoutCache.size,
  };
}
