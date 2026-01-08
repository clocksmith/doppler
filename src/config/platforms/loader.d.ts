/**
 * Platform Loader
 *
 * Detects the current GPU platform and loads appropriate configs.
 * Provides platform-specific kernel overrides and memory hints.
 *
 * @module config/platforms/loader
 */

import type {
  PlatformSchema,
  RuntimeCapabilities,
  ResolvedPlatformConfig,
  KernelOperationOverrideSchema,
  MemoryHintsSchema,
} from '../schema/platform.schema.js';

/**
 * Set the base URL for loading platform configs.
 */
export function setPlatformsBaseUrl(baseUrl: string): void;

/**
 * Detect platform from WebGPU adapter info.
 */
export function detectPlatform(adapterInfo: GPUAdapterInfo): Promise<PlatformSchema>;

/**
 * Initialize platform detection with a WebGPU adapter.
 */
export function initializePlatform(adapter: GPUAdapter): Promise<ResolvedPlatformConfig>;

/**
 * Get the current platform (throws if not initialized).
 */
export function getPlatform(): PlatformSchema;

/**
 * Get the current runtime capabilities (throws if not initialized).
 */
export function getCapabilities(): RuntimeCapabilities;

/**
 * Get kernel override for an operation on current platform.
 */
export function getKernelOverride(operation: string): KernelOperationOverrideSchema | undefined;

/**
 * Get preferred variant for an operation, if platform specifies one.
 */
export function getPreferredVariant(operation: string): string | undefined;

/**
 * Check if a variant should be avoided on current platform.
 */
export function shouldAvoidVariant(operation: string, variant: string): boolean;

/**
 * Get workgroup size override for a variant, if platform specifies one.
 */
export function getWorkgroupOverride(
  operation: string,
  variant: string
): [number, number, number] | undefined;

/**
 * Get WGSL override constants for a variant, if platform specifies any.
 */
export function getWgslOverrides(
  operation: string,
  variant: string
): Record<string, number> | undefined;

/**
 * Get memory hints for current platform.
 */
export function getMemoryHints(): MemoryHintsSchema | undefined;

/**
 * Check if current platform prefers unified memory strategies.
 */
export function prefersUnifiedMemory(): boolean;

/**
 * Get optimal buffer alignment for current platform.
 */
export function getBufferAlignment(): number;

/**
 * Clear all cached platform data. Useful for hot-reloading.
 */
export function clearPlatformCache(): void;

/**
 * Get resolved platform config with capabilities.
 */
export function getResolvedPlatformConfig(): ResolvedPlatformConfig;
