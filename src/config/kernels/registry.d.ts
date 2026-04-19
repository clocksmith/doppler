/**
 * Kernel Registry Loader
 *
 * Loads and caches the kernel registry from JSON.
 * Provides resolved kernel configs with base + variant merged.
 *
 * @module config/kernels/registry
 */

import type {
  KernelRegistrySchema,
  OperationSchema,
  KernelVariantSchema,
  BindingSchema,
  ResolvedKernelConfig,
} from '../schema/kernel-registry.schema.js';
import type { RuntimeCapabilities } from '../schema/platform.schema.js';

/**
 * Set the URL for loading the registry.
 * Must be called before getRegistry() if not using default.
 */
export function setRegistryUrl(url: string): void;

/**
 * Get the kernel registry, loading it if needed.
 */
export function getRegistry(): Promise<KernelRegistrySchema>;

/**
 * Clear the cached registry. Useful for hot-reloading.
 */
export function clearRegistryCache(): void;

/**
 * Merge base and variant bindings.
 * Variant bindings with matching indices override base bindings.
 */
export function mergeBindings(
  base: BindingSchema[],
  override: BindingSchema[] | undefined
): BindingSchema[];

/**
 * Resolve a kernel variant to a complete configuration.
 * Merges base operation config with variant-specific overrides.
 */
export function resolveKernelConfig(
  operation: string,
  variant: string
): ResolvedKernelConfig | null;
