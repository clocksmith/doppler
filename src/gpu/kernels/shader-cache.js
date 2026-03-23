

import { log } from '../../debug/index.js';
import { getDeviceEpoch } from '../device.js';

// ============================================================================
// Caches
// ============================================================================

const MAX_SHADER_SOURCE_CACHE_SIZE = 256;
const MAX_SHADER_MODULE_CACHE_SIZE = 256;

// Map maintains insertion order; eviction deletes the oldest (first) key.
const shaderSourceCache = new Map();

const shaderModuleCache = new Map();

function evictOldest(map, maxSize) {
  while (map.size > maxSize) {
    const oldest = map.keys().next().value;
    map.delete(oldest);
  }
}

function touchCacheEntry(map, key, value) {
  // Move to end for LRU ordering: delete then re-insert.
  map.delete(key);
  map.set(key, value);
}

let moduleCacheEpoch = -1;
const deviceIds = new WeakMap();
let nextDeviceId = 1;

function nodeModule(specifier) {
  return `node:${specifier}`;
}

function getDeviceId(device) {
  let id = deviceIds.get(device);
  if (id == null) {
    id = nextDeviceId++;
    deviceIds.set(device, id);
  }
  return id;
}

function ensureModuleCacheEpoch() {
  const epoch = getDeviceEpoch();
  if (epoch !== moduleCacheEpoch) {
    shaderModuleCache.clear();
    moduleCacheEpoch = epoch;
  }
}

// ============================================================================
// Base Path Detection
// ============================================================================


function getKernelBasePath() {
  // Allow an app to override shader base path without rebuilding Doppler.
  // Dream uses this when serving Doppler sources from /reploid/doppler/... instead of /src/...
  const override = (typeof globalThis !== 'undefined') ? globalThis.__DOPPLER_KERNEL_BASE_PATH__ : null;
  if (typeof override === 'string' && override.trim()) {
    return override.replace(/\/+$/, '');
  }

  // Node runtimes do not have location; resolve kernels relative to this module.
  if (typeof location === 'undefined') {
    return new URL('.', import.meta.url).toString().replace(/\/$/, '');
  }

  // Check common deployed Doppler paths.
  if (typeof location !== 'undefined') {
    const path = location.pathname || '';
    if (
      path === '/d' ||
      path.startsWith('/d/') ||
      path === '/doppler' ||
      path.startsWith('/doppler/') ||
      path === '/dr' ||
      path.startsWith('/dr/') ||
      location.host.includes('replo')
    ) {
      return '/doppler/src/gpu/kernels';
    }
  }
  return '/src/gpu/kernels';
}

const KERNEL_BASE_PATH = getKernelBasePath();

function isFileUrl(value) {
  return typeof value === 'string' && value.startsWith('file://');
}

async function loadShaderSourceFromFileUrl(url) {
  try {
    const fs = await import(nodeModule('fs/promises'));
    const source = await fs.readFile(new URL(url), 'utf8');
    return source;
  } catch (error) {
    log.error('ShaderCache', `Failed to read shader via file URL ${url}: ${error}`);
    throw error;
  }
}

// ============================================================================
// Shader Loading
// ============================================================================


export async function loadShaderSource(filename) {
  if (shaderSourceCache.has(filename)) {
    const cached = shaderSourceCache.get(filename);
    touchCacheEntry(shaderSourceCache, filename, cached);
    return cached;
  }

  const url = `${KERNEL_BASE_PATH}/${filename}`;
  try {
    // Node's fetch may not support file:// URLs; use fs fallback for that path.
    if (isFileUrl(url)) {
      const source = await loadShaderSourceFromFileUrl(url);
      shaderSourceCache.set(filename, source);
      evictOldest(shaderSourceCache, MAX_SHADER_SOURCE_CACHE_SIZE);
      return source;
    }

    const response = await fetch(url, { cache: 'no-cache' });
    if (!response.ok) {
      throw new Error(`Failed to load shader ${filename}: ${response.status}`);
    }
    const source = await response.text();
    shaderSourceCache.set(filename, source);
    evictOldest(shaderSourceCache, MAX_SHADER_SOURCE_CACHE_SIZE);
    return source;
  } catch (error) {
    log.error('ShaderCache', `Failed to load shader ${filename}: ${error}`);
    throw error;
  }
}

// ============================================================================
// Shader Compilation
// ============================================================================


export async function compileShader(
  device,
  source,
  label
) {
  let module;
  try {
    module = device.createShaderModule({
      label,
      code: source,
    });
  } catch (err) {
    throw new Error(`createShaderModule failed for "${label}": ${err.message}`);
  }

  // Check for compilation errors (getCompilationInfo not available in all WebGPU providers)
  const compilationInfo = typeof module.getCompilationInfo === 'function'
    ? await module.getCompilationInfo()
    : { messages: [] };
  if (compilationInfo.messages.length > 0) {
    for (const msg of compilationInfo.messages) {
      if (msg.type === 'error') {
        log.error('compileShader', `${label}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      } else if (msg.type === 'warning') {
        log.warn('compileShader', `${label}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      } else {
        log.debug('compileShader', `${label}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      }
    }
    if (compilationInfo.messages.some(m => m.type === 'error')) {
      throw new Error(`Shader compilation failed for ${label}`);
    }
  }

  return module;
}


export async function getShaderModule(
  device,
  shaderFile,
  label
) {
  ensureModuleCacheEpoch();
  const cacheKey = `${getDeviceId(device)}:${shaderFile}`;
  const cached = shaderModuleCache.get(cacheKey);
  if (cached) {
    touchCacheEntry(shaderModuleCache, cacheKey, cached);
    return cached;
  }

  const compilePromise = (async () => {
    const shaderSource = await loadShaderSource(shaderFile);
    return compileShader(device, shaderSource, label);
  })();

  shaderModuleCache.set(cacheKey, compilePromise);
  evictOldest(shaderModuleCache, MAX_SHADER_MODULE_CACHE_SIZE);

  try {
    return await compilePromise;
  } catch (err) {
    shaderModuleCache.delete(cacheKey);
    throw err;
  }
}

// ============================================================================
// Cache Management
// ============================================================================


export function clearShaderCaches() {
  shaderSourceCache.clear();
  shaderModuleCache.clear();
  moduleCacheEpoch = getDeviceEpoch();
}


export function getShaderCacheStats() {
  return {
    sources: shaderSourceCache.size,
    modules: shaderModuleCache.size,
  };
}
