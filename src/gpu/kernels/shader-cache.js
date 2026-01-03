

import { log } from '../../debug/index.js';
import { getDeviceEpoch } from '../device.js';

// ============================================================================
// Caches
// ============================================================================


const shaderSourceCache = new Map();


const shaderModuleCache = new Map();

let moduleCacheEpoch = -1;
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

// ============================================================================
// Shader Loading
// ============================================================================


export async function loadShaderSource(filename) {
  if (shaderSourceCache.has(filename)) {
    return shaderSourceCache.get(filename);
  }

  const url = `${KERNEL_BASE_PATH}/${filename}`;
  try {
    const response = await fetch(url, { cache: 'no-cache' });
    if (!response.ok) {
      throw new Error(`Failed to load shader ${filename}: ${response.status}`);
    }
    const source = await response.text();
    shaderSourceCache.set(filename, source);
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
  const module = device.createShaderModule({
    label,
    code: source,
  });

  // Check for compilation errors
  const compilationInfo = await module.getCompilationInfo();
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
    return cached;
  }

  const compilePromise = (async () => {
    const shaderSource = await loadShaderSource(shaderFile);
    return compileShader(device, shaderSource, label);
  })();

  shaderModuleCache.set(cacheKey, compilePromise);

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
