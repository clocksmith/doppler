

import { log } from '../../debug/index.js';
import { isDeviceLost, observeDeviceLoss } from '../device-state.js';
import { getScopedShaderSource } from './shader-source-scope.js';
import { hashBytesSha256 } from '../../formats/canonical-hash.js';

// ============================================================================
// Caches
// ============================================================================

const MAX_SHADER_SOURCE_CACHE_SIZE = 256;
const MAX_SHADER_MODULE_CACHE_SIZE = 256;

// Map maintains insertion order; eviction deletes the oldest (first) key.
const shaderSourceCache = new Map();

let shaderModuleCaches = new WeakMap();
let shaderModuleCount = 0;
const sourceIdentities = new Map();
const moduleIdentities = new WeakMap();

function sourceIdentity(filename, source) {
  const cached = sourceIdentities.get(filename);
  if (cached?.source === source) return cached.digest;
  const digest = hashBytesSha256(new TextEncoder().encode(source));
  sourceIdentities.set(filename, { source, digest });
  evictOldest(sourceIdentities, MAX_SHADER_SOURCE_CACHE_SIZE);
  return digest;
}

export function getShaderSourceIdentity(filename) {
  const scoped = getScopedShaderSource(filename);
  if (scoped) return scoped.digest;
  const source = shaderSourceCache.get(filename) ?? shaderSourcePreseeds.get(filename);
  return source === undefined ? null : sourceIdentity(filename, source);
}

export function getShaderModuleIdentity(module) {
  return moduleIdentities.get(module);
}

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

function moduleCacheFor(device) {
  observeDeviceLoss(device);
  if (!device || isDeviceLost(device)) throw new Error('Shader cache requires a live GPU device.');
  let cache = shaderModuleCaches.get(device);
  if (cache) return cache;
  cache = new Map();
  shaderModuleCaches.set(device, cache);
  const caches = shaderModuleCaches;
  const clear = () => {
    if (caches === shaderModuleCaches) shaderModuleCount -= cache.size;
    cache.clear();
    caches.delete(device);
  };
  device.lost?.then(clear, clear);
  return cache;
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

  return new URL('.', import.meta.url).toString().replace(/\/$/, '');
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


// Consumer-preseeded shader sources. Bundlers that can resolve
// `import.meta.glob('./kernels/*.wgsl', { as: 'raw', eager: true })` or
// equivalent should call `registerShaderSources(map)` before any pipeline
// init; the runtime skips the fetch path entirely when a preseed is present.
const shaderSourcePreseeds = new Map();

export function registerShaderSources(map) {
  if (!map) return;
  const entries = map instanceof Map ? map.entries() : Object.entries(map);
  for (const [key, value] of entries) {
    if (typeof value !== 'string') continue;
    const filename = String(key).split('/').pop();
    if (!filename) continue;
    shaderSourcePreseeds.set(filename, value);
    shaderSourceCache.delete(filename);
  }
}

export function hasPreseededShaderSource(filename) {
  return shaderSourcePreseeds.has(filename);
}

export async function loadShaderSource(filename) {
  const scoped = getScopedShaderSource(filename);
  if (scoped) return scoped.source;
  if (shaderSourceCache.has(filename)) {
    const cached = shaderSourceCache.get(filename);
    touchCacheEntry(shaderSourceCache, filename, cached);
    return cached;
  }
  const preseeded = shaderSourcePreseeds.get(filename);
  if (typeof preseeded === 'string') {
    shaderSourceCache.set(filename, preseeded);
    evictOldest(shaderSourceCache, MAX_SHADER_SOURCE_CACHE_SIZE);
    return preseeded;
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
  const shaderModuleCache = moduleCacheFor(device);
  const shaderSource = await loadShaderSource(shaderFile);
  const digest = sourceIdentity(shaderFile, shaderSource);
  const cacheKey = `${getDeviceId(device)}:${digest}`;
  const cached = shaderModuleCache.get(cacheKey);
  if (cached) {
    touchCacheEntry(shaderModuleCache, cacheKey, cached);
    return cached;
  }

  const compilePromise = (async () => {
    const module = await compileShader(device, shaderSource, label);
    moduleIdentities.set(module, digest);
    return module;
  })();

  shaderModuleCache.set(cacheKey, compilePromise);
  const beforeEviction = shaderModuleCache.size;
  evictOldest(shaderModuleCache, MAX_SHADER_MODULE_CACHE_SIZE);
  shaderModuleCount += 1 - (beforeEviction - shaderModuleCache.size);

  try {
    const module = await compilePromise;
    if (isDeviceLost(device)) throw new Error('Device lost while compiling shader.');
    return module;
  } catch (err) {
    if (shaderModuleCache.get(cacheKey) === compilePromise && shaderModuleCache.delete(cacheKey)
      && shaderModuleCaches.get(device) === shaderModuleCache) shaderModuleCount--;
    throw err;
  }
}

// ============================================================================
// Cache Management
// ============================================================================


export function clearShaderCaches() {
  shaderSourceCache.clear();
  shaderModuleCaches = new WeakMap();
  shaderModuleCount = 0;
  sourceIdentities.clear();
}


export function getShaderCacheStats(device = null) {
  return {
    sources: shaderSourceCache.size,
    modules: device ? (shaderModuleCaches.get(device)?.size ?? 0) : shaderModuleCount,
  };
}
