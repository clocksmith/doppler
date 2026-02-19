

import { getDevice, getDeviceEpoch, getKernelCapabilities } from '../device.js';
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

let pipelineCacheEpoch = -1;
const deviceIds = new WeakMap();
let nextDeviceId = 1;

function getDeviceId(device) {
  let id = deviceIds.get(device);
  if (id == null) {
    id = nextDeviceId++;
    deviceIds.set(device, id);
  }
  return id;
}

function ensureCacheEpoch() {
  const epoch = getDeviceEpoch();
  if (epoch !== pipelineCacheEpoch) {
    pipelineCache.clear();
    bindGroupLayoutCache.clear();
    pipelineLayoutCache.clear();
    pipelineCacheEpoch = epoch;
  }
}

// ============================================================================
// Bind Group Layout
// ============================================================================


export function getOrCreateBindGroupLayout(
  label,
  entries,
  deviceOverride = null
) {
  ensureCacheEpoch();

  const device = deviceOverride || getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }
  const scopedLabel = `${getDeviceId(device)}:${label}`;
  const cached = bindGroupLayoutCache.get(scopedLabel);
  if (cached) {
    return cached;
  }

  const layout = device.createBindGroupLayout({ label, entries });
  bindGroupLayoutCache.set(scopedLabel, layout);
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
  ensureCacheEpoch();

  const device = deviceOverride || getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }
  const scopedLabel = `${getDeviceId(device)}:${label}`;
  const cached = pipelineLayoutCache.get(scopedLabel);
  if (cached) {
    return cached;
  }

  const layout = device.createPipelineLayout({
    label,
    bindGroupLayouts,
  });

  pipelineLayoutCache.set(scopedLabel, layout);
  return layout;
}

// ============================================================================
// Pipeline Creation
// ============================================================================


function buildPipelineCacheKey(operation, variant, constants, bindGroupLayout, device) {
  const constantsKey = constants
    ? Object.entries(constants).sort().map(([k, v]) => `${k}=${v}`).join('|')
    : '';
  const layoutKey = bindGroupLayout ? `:${bindGroupLayout.label || 'layout'}` : '';
  const deviceKey = `dev:${getDeviceId(device)}`;
  return `${deviceKey}:${operation}:${variant}${constants ? ':' + constantsKey : ''}${layoutKey}`;
}

function resolveConstants(operation, variant, constants) {
  const config = getKernelConfig(operation, variant);
  const overrides = config.wgslOverrides;
  if (!overrides || Object.keys(overrides).length === 0) {
    return constants;
  }
  if (!constants || Object.keys(constants).length === 0) {
    return { ...overrides };
  }
  return { ...constants, ...overrides };
}

function isPowerOfTwo(value) {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

function assertWorkgroupPowerOfTwo(operation, variant, workgroupSize, constants) {
  if (Array.isArray(workgroupSize)) {
    for (const dim of workgroupSize) {
      if (!isPowerOfTwo(dim)) {
        throw new Error(
          `Kernel ${operation}/${variant} requires power-of-two workgroup size, got [${workgroupSize.join(', ')}]`
        );
      }
    }
  }

  if (!constants) return;
  for (const [key, value] of Object.entries(constants)) {
    if (!Number.isFinite(value)) continue;
    if (!key.includes('WORKGROUP_SIZE')) continue;
    if (!isPowerOfTwo(value)) {
      throw new Error(
        `Kernel ${operation}/${variant} requires power-of-two ${key}, got ${value}`
      );
    }
  }
}

export function getCachedPipeline(
  operation,
  variant,
  constants = null
) {
  ensureCacheEpoch();
  const device = getDevice();
  if (!device) {
    return null;
  }
  const resolvedConstants = resolveConstants(operation, variant, constants);
  const cacheKey = buildPipelineCacheKey(operation, variant, resolvedConstants, null, device);
  return pipelineCache.get(cacheKey) || null;
}


export async function getPipelineFast(
  operation,
  variant,
  bindGroupLayout = null,
  constants = null
) {
  ensureCacheEpoch();
  const device = getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }
  const resolvedConstants = resolveConstants(operation, variant, constants);
  if (bindGroupLayout) {
    const layoutKey = buildPipelineCacheKey(operation, variant, resolvedConstants, bindGroupLayout, device);
    const cached = pipelineCache.get(layoutKey);
    if (cached) return cached;
    return createPipeline(operation, variant, bindGroupLayout, constants);
  }
  const cacheKey = buildPipelineCacheKey(operation, variant, resolvedConstants, null, device);
  const cached = pipelineCache.get(cacheKey);
  if (cached) return cached;
  return createPipeline(operation, variant, null, constants);
}


export async function createPipeline(
  operation,
  variant,
  bindGroupLayout = null,
  constants = null
) {
  ensureCacheEpoch();
  const device = getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }

  const config = getKernelConfig(operation, variant);
  const resolvedConstants = resolveConstants(operation, variant, constants);
  const constantsKey = resolvedConstants
    ? Object.entries(resolvedConstants).sort().map(([k, v]) => `${k}=${v}`).join('|')
    : '';
  const cacheKey = buildPipelineCacheKey(operation, variant, resolvedConstants, bindGroupLayout, device);

  // Return cached pipeline if available
  if (pipelineCache.has(cacheKey)) {
    return pipelineCache.get(cacheKey);
  }
  const capabilities = getKernelCapabilities();

  // Verify requirements
  if (!hasRequiredFeatures(config.requires, capabilities)) {
    throw new Error(
      `Kernel ${operation}/${variant} requires features: ${config.requires.join(', ')}`
    );
  }

  assertWorkgroupPowerOfTwo(operation, variant, config.workgroupSize, resolvedConstants);

  trace.kernels(
    `KernelLayout: ${operation}/${variant} file=${config.shaderFile} entry=${config.entryPoint} ` +
    `workgroup=[${config.workgroupSize.join(',')}] requires=` +
    `${config.requires.length > 0 ? config.requires.join('|') : 'none'}` +
    `${resolvedConstants ? ' constants=' + constantsKey : ''}`
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
      constants: resolvedConstants || undefined,
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
  pipelineCacheEpoch = getDeviceEpoch();
}


export function getPipelineCacheStats() {
  return {
    pipelines: pipelineCache.size,
    bindGroupLayouts: bindGroupLayoutCache.size,
    pipelineLayouts: pipelineLayoutCache.size,
  };
}
