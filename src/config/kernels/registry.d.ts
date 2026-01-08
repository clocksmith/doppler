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
 * Get registry synchronously (throws if not loaded).
 * Use after awaiting getRegistry() at startup.
 */
export function getRegistrySync(): KernelRegistrySchema;

/**
 * Clear the cached registry. Useful for hot-reloading.
 */
export function clearRegistryCache(): void;

/**
 * Get an operation schema by name.
 */
export function getOperation(operation: string): OperationSchema | undefined;

/**
 * Get a variant schema by operation and variant name.
 */
export function getVariant(operation: string, variant: string): KernelVariantSchema | undefined;

/**
 * Get all variant names for an operation.
 */
export function getVariantNames(operation: string): string[];

/**
 * Check if a variant's requirements are met by capabilities.
 */
export function isVariantAvailable(
  operation: string,
  variant: string,
  capabilities: RuntimeCapabilities
): boolean;

/**
 * Get all available variants for an operation given capabilities.
 */
export function getAvailableVariants(operation: string, capabilities: RuntimeCapabilities): string[];

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
