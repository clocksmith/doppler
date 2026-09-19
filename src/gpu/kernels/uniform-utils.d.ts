/**
 * Uniform Utils - Uniform buffer creation utilities
 *
 * Provides utilities for creating and caching uniform buffers
 * for kernel dispatch.
 *
 * @module gpu/kernels/uniform-utils
 */

import type { CommandRecorder } from '../command-recorder.js';
import type { KernelConfig } from '../../config/kernel-registry-contract.js';

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
 * Write a kernel's uniform fields into a DataView according to the
 * registry-declared layout. Throws when the kernel has no uniforms or
 * a required field is absent or outside its binary type's range. Only declared
 * padding is filled with zero. Additional values shared across variants are ignored.
 */
export declare function writeUniformsFromObject(
  view: DataView,
  config: KernelConfig,
  values: Record<string, number>
): void;

/**
 * Byte length required for a kernel's uniform buffer, taking the
 * declared size and alignment rules into account. Returns 0 when the
 * kernel has no uniforms.
 */
export declare function getUniformByteLength(config: KernelConfig): number;

export declare function createKernelUniformBuffer(
  label: string,
  config: KernelConfig,
  values: Record<string, number>,
  recorder?: CommandRecorder | null,
  deviceOverride?: GPUDevice | null
): GPUBuffer;

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
