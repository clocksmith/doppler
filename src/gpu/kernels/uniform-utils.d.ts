/**
 * Uniform Utils - Uniform buffer creation utilities
 *
 * Provides utilities for creating and caching uniform buffers
 * for kernel dispatch.
 *
 * @module gpu/kernels/uniform-utils
 */

import type { CommandRecorder } from '../command-recorder.js';

// ============================================================================
// Types
// ============================================================================

/** Options for uniform buffer creation */
export interface UniformBufferOptions {
  /** Use content-addressed cache for reuse (default: true) */
  useCache?: boolean;
}

// ============================================================================
// Uniform Buffer Creation
// ============================================================================

/**
 * Create a uniform buffer from raw data.
 * Uses caching by default for content-addressed reuse.
 */
export declare function createUniformBufferFromData(
  label: string,
  data: ArrayBuffer | ArrayBufferView,
  recorder?: CommandRecorder | null,
  deviceOverride?: GPUDevice | null,
  options?: UniformBufferOptions
): GPUBuffer;

/**
 * Create a uniform buffer with a DataView writer callback.
 * Allows structured data writing with proper alignment.
 */
export declare function createUniformBufferWithView(
  label: string,
  byteLength: number,
  writer: (view: DataView) => void,
  recorder?: CommandRecorder | null,
  deviceOverride?: GPUDevice | null
): GPUBuffer;
