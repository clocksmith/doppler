

import { getDevice } from '../device.js';
import { isDeviceLost, observeDeviceLoss } from '../device-state.js';
import { getKernelConfig } from './kernel-configs.js';
import { getShaderModule } from './shader-cache.js';
import { getScopedShaderSource } from './shader-source-scope.js';
import { hasRequiredFeatures } from './feature-check.js';
import { trace } from '../../debug/index.js';

// ============================================================================
// Caches
// ============================================================================


let deviceCaches = new WeakMap();

let pipelineBindGroupLayoutCache = new WeakMap();

const deviceIds = new WeakMap();
let nextDeviceId = 1;
const layoutIds = new WeakMap();
let nextLayoutId = 1;

function getLayoutId(layout) {
  if (!layoutIds.has(layout)) layoutIds.set(layout, nextLayoutId++);
  return layoutIds.get(layout);
}

function getDeviceId(device) {
  let id = deviceIds.get(device);
  if (id == null) {
    id = nextDeviceId++;
    deviceIds.set(device, id);
  }
  return id;
}

function getDeviceCaches(device) {
  observeDeviceLoss(device);
  if (!device || isDeviceLost(device)) throw new Error('Pipeline cache requires a live GPU device.');
  let caches = deviceCaches.get(device);
  if (!caches) {
    caches = { pipelines: new Map(), bindGroupLayouts: new Map(), pipelineLayouts: new Map() };
    deviceCaches.set(device, caches);
    device.lost?.then(() => deviceCaches.delete(device), () => deviceCaches.delete(device));
  }
  return caches;
}

// ============================================================================
// Bind Group Layout
// ============================================================================


export function getOrCreateBindGroupLayout(
  label,
  entries,
  deviceOverride = null
) {
  const device = deviceOverride || getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }
  const scopedLabel = `${getDeviceId(device)}:${label}:${JSON.stringify(entries)}`;
  const bindGroupLayoutCache = getDeviceCaches(device).bindGroupLayouts;
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
  const device = deviceOverride || getDevice();
  if (!device) {
    throw new Error('Device not initialized');
  }
  const scopedLabel = `${getDeviceId(device)}:${label}:${bindGroupLayouts.map(getLayoutId).join(',')}`;
  const pipelineLayoutCache = getDeviceCaches(device).pipelineLayouts;
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

export function getPipelineBindGroupLayout(pipeline, index = 0) {
  if (!pipeline || typeof pipeline.getBindGroupLayout !== 'function') {
    throw new Error('getPipelineBindGroupLayout requires a GPUComputePipeline');
  }
  const normalizedIndex = Number.isInteger(index) && index >= 0 ? index : 0;
  let layoutsByIndex = pipelineBindGroupLayoutCache.get(pipeline);
  if (!layoutsByIndex) {
    layoutsByIndex = new Map();
    pipelineBindGroupLayoutCache.set(pipeline, layoutsByIndex);
  }
  const cached = layoutsByIndex.get(normalizedIndex);
  if (cached) {
    return cached;
  }
  const layout = pipeline.getBindGroupLayout(normalizedIndex);
  layoutsByIndex.set(normalizedIndex, layout);
  return layout;
}

// ============================================================================
// Pipeline Creation
// ============================================================================


function buildPipelineCacheKey(operation, variant, constants, bindGroupLayout, device) {
  const constantsKey = constants
    ? Object.entries(constants).sort().map(([k, v]) => `${k}=${v}`).join('|')
    : '';
  const layoutKey = bindGroupLayout ? `:layout:${getLayoutId(bindGroupLayout)}` : '';
  const deviceKey = `dev:${getDeviceId(device)}`;
  const source = getScopedShaderSource(getKernelConfig(operation, variant).shaderFile);
  return `${deviceKey}:${source?.digest ?? 'runtime'}:${operation}:${variant}${constants ? ':' + constantsKey : ''}${layoutKey}`;
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

function normalizePipelineConstants(constants) {
  if (!constants) return null;
  const normalized = {};
  for (const [key, value] of Object.entries(constants)) {
    if (typeof value === 'boolean') {
      normalized[key] = value ? 1 : 0;
      continue;
    }
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new Error(
        `Kernel pipeline constant "${key}" must be a finite number or boolean, got ${typeof value}.`
      );
    }
    normalized[key] = value;
  }
  return normalized;
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
  constants = null,
  device = getDevice()
) {
  if (!device) {
    return null;
  }
  const resolvedConstants = normalizePipelineConstants(
    resolveConstants(operation, variant, constants)
  );
  const cacheKey = buildPipelineCacheKey(operation, variant, resolvedConstants, null, device);
  const pipelineCache = getDeviceCaches(device).pipelines;
  return pipelineCache.get(cacheKey) || null;
}


export async function getPipelineFast(
  operation,
  variant,
  bindGroupLayout = null,
  constants = null,
  device = getDevice()
) {
  if (!device) {
    throw new Error('Device not initialized');
  }
  const resolvedConstants = normalizePipelineConstants(
    resolveConstants(operation, variant, constants)
  );
  const pipelineCache = getDeviceCaches(device).pipelines;
  if (bindGroupLayout) {
    const layoutKey = buildPipelineCacheKey(operation, variant, resolvedConstants, bindGroupLayout, device);
    const cached = pipelineCache.get(layoutKey);
    if (cached) return cached;
    return createPipeline(operation, variant, bindGroupLayout, constants, device);
  }
  const cacheKey = buildPipelineCacheKey(operation, variant, resolvedConstants, null, device);
  const cached = pipelineCache.get(cacheKey);
  if (cached) return cached;
  return createPipeline(operation, variant, null, constants, device);
}


export async function createPipeline(
  operation,
  variant,
  bindGroupLayout = null,
  constants = null,
  device = getDevice()
) {
  if (!device) {
    throw new Error('Device not initialized');
  }

  const config = getKernelConfig(operation, variant);
  const pipelineCache = getDeviceCaches(device).pipelines;
  const resolvedConstants = normalizePipelineConstants(
    resolveConstants(operation, variant, constants)
  );
  const constantsKey = resolvedConstants
    ? Object.entries(resolvedConstants).sort().map(([k, v]) => `${k}=${v}`).join('|')
    : '';
  const cacheKey = buildPipelineCacheKey(operation, variant, resolvedConstants, bindGroupLayout, device);

  // Return cached pipeline if available
  if (pipelineCache.has(cacheKey)) {
    return pipelineCache.get(cacheKey);
  }
  const capabilities = {
    hasF16: device.features.has('shader-f16'),
    hasSubgroups: device.features.has('subgroups'),
  };

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
  if (isDeviceLost(device)) throw new Error('GPU device lost while creating a pipeline.');
  pipelineCache.set(cacheKey, pipeline);

  return pipeline;
}

// ============================================================================
// Cache Management
// ============================================================================


export function clearPipelineCaches() {
  deviceCaches = new WeakMap();
  pipelineBindGroupLayoutCache = new WeakMap();
}


export function getPipelineCacheStats(device = getDevice()) {
  const caches = device ? deviceCaches.get(device) : null;
  return {
    pipelines: caches?.pipelines.size ?? 0,
    bindGroupLayouts: caches?.bindGroupLayouts.size ?? 0,
    pipelineLayouts: caches?.pipelineLayouts.size ?? 0,
  };
}
