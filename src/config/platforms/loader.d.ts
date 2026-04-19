/**
 * Platform Loader
 *
 * Detects the current GPU platform and loads appropriate configs.
 *
 * @module config/platforms/loader
 */

import type {
  PlatformSchema,
  RuntimeCapabilities,
  ResolvedPlatformConfig,
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
 * Clear all cached platform data. Useful for hot-reloading.
 */
export function clearPlatformCache(): void;
