/**
 * Shader Cache - Shader loading and compilation utilities
 *
 * Handles loading WGSL shader sources from disk/network and compiling
 * them into GPUShaderModules with caching.
 *
 * @module gpu/kernels/shader-cache
 */

// ============================================================================
// Shader Loading
// ============================================================================

/**
 * Load a WGSL shader file via fetch
 */
export declare function loadShaderSource(filename: string): Promise<string>;

// ============================================================================
// Shader Compilation
// ============================================================================

/**
 * Compile a shader module
 */
export declare function compileShader(
  device: GPUDevice,
  source: string,
  label: string
): Promise<GPUShaderModule>;

/**
 * Get or create a cached shader module for a shader file.
 */
export declare function getShaderModule(
  device: GPUDevice,
  shaderFile: string,
  label: string
): Promise<GPUShaderModule>;

// ============================================================================
// Cache Management
// ============================================================================

/**
 * Clear the shader caches
 */
export declare function clearShaderCaches(): void;

/**
 * Get shader cache statistics
 */
export declare function getShaderCacheStats(): {
  sources: number;
  modules: number;
};
