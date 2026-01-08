/**
 * Shader Cache - Shader loading and compilation utilities
 *
 * Handles loading WGSL shader sources from disk/network and compiling
 * them into GPUShaderModules with caching.
 *
 * @module gpu/kernels/shader-cache
 */

import { log } from '../../debug/index.js';

// ============================================================================
// Caches
// ============================================================================

/** Shader source cache (loaded via fetch) */
const shaderSourceCache = new Map<string, string>();

/** Compiled shader module cache */
const shaderModuleCache = new Map<string, Promise<GPUShaderModule>>();

// ============================================================================
// Base Path Detection
// ============================================================================

/**
 * Base path for kernel files
 * Detects if running under /doppler/ (replo.id deployment) or standalone
 */
function getKernelBasePath(): string {
  // Check if we're running from /doppler/ path (replo.id deployment)
  if (typeof location !== 'undefined') {
    const path = location.pathname;
    if (path.startsWith('/d') || path.startsWith('/doppler/') || location.host.includes('replo')) {
      return '/doppler/gpu/kernels';
    }
  }
  return '/gpu/kernels';
}

const KERNEL_BASE_PATH = getKernelBasePath();

// ============================================================================
// Shader Loading
// ============================================================================

/**
 * Load a WGSL shader file via fetch
 */
export async function loadShaderSource(filename: string): Promise<string> {
  if (shaderSourceCache.has(filename)) {
    return shaderSourceCache.get(filename)!;
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

/**
 * Compile a shader module
 */
export async function compileShader(
  device: GPUDevice,
  source: string,
  label: string
): Promise<GPUShaderModule> {
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

/**
 * Get or create a cached shader module for a shader file.
 */
export async function getShaderModule(
  device: GPUDevice,
  shaderFile: string,
  label: string
): Promise<GPUShaderModule> {
  const cacheKey = shaderFile;
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

/**
 * Clear the shader caches
 */
export function clearShaderCaches(): void {
  shaderSourceCache.clear();
  shaderModuleCache.clear();
}

/**
 * Get shader cache statistics
 */
export function getShaderCacheStats(): {
  sources: number;
  modules: number;
} {
  return {
    sources: shaderSourceCache.size,
    modules: shaderModuleCache.size,
  };
}
