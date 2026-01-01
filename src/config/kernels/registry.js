/**
 * Kernel Registry Loader
 *
 * Loads and caches the kernel registry from JSON.
 * Provides resolved kernel configs with base + variant merged.
 *
 * @module config/kernels/registry
 */

/** @type {import('../schema/kernel-registry.schema.js').KernelRegistrySchema | null} */
let cachedRegistry = null;

/** @type {string | null} */
let registryUrl = null;

/**
 * Set the URL for loading the registry.
 * Must be called before getRegistry() if not using default.
 * @param {string} url
 */
export function setRegistryUrl(url) {
  registryUrl = url;
  cachedRegistry = null; // Clear cache when URL changes
}

/**
 * Get the kernel registry, loading it if needed.
 * @returns {Promise<import('../schema/kernel-registry.schema.js').KernelRegistrySchema>}
 */
export async function getRegistry() {
  if (cachedRegistry) {
    return cachedRegistry;
  }

  const url = registryUrl || new URL('./registry.json', import.meta.url).href;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load kernel registry from ${url}: ${response.status}`);
  }

  cachedRegistry = await response.json();
  return cachedRegistry;
}

/**
 * Get registry synchronously (throws if not loaded).
 * Use after awaiting getRegistry() at startup.
 * @returns {import('../schema/kernel-registry.schema.js').KernelRegistrySchema}
 */
export function getRegistrySync() {
  if (!cachedRegistry) {
    throw new Error('Kernel registry not loaded. Call await getRegistry() first.');
  }
  return cachedRegistry;
}

/**
 * Clear the cached registry. Useful for hot-reloading.
 */
export function clearRegistryCache() {
  cachedRegistry = null;
}

/**
 * Get an operation schema by name.
 * @param {string} operation
 * @returns {import('../schema/kernel-registry.schema.js').OperationSchema | undefined}
 */
export function getOperation(operation) {
  const registry = getRegistrySync();
  return registry.operations[operation];
}

/**
 * Get a variant schema by operation and variant name.
 * @param {string} operation
 * @param {string} variant
 * @returns {import('../schema/kernel-registry.schema.js').KernelVariantSchema | undefined}
 */
export function getVariant(operation, variant) {
  const op = getOperation(operation);
  return op?.variants[variant];
}

/**
 * Get all variant names for an operation.
 * @param {string} operation
 * @returns {string[]}
 */
export function getVariantNames(operation) {
  const op = getOperation(operation);
  return op ? Object.keys(op.variants) : [];
}

/**
 * Check if a variant's requirements are met by capabilities.
 * @param {string} operation
 * @param {string} variant
 * @param {import('../schema/platform.schema.js').RuntimeCapabilities} capabilities
 * @returns {boolean}
 */
export function isVariantAvailable(operation, variant, capabilities) {
  const variantSchema = getVariant(operation, variant);
  if (!variantSchema) return false;

  const requires = variantSchema.requires || [];
  for (const req of requires) {
    if (req === 'shader-f16' && !capabilities.hasF16) return false;
    if (req === 'subgroups' && !capabilities.hasSubgroups) return false;
    if (req === 'subgroups-f16' && (!capabilities.hasSubgroups || !capabilities.hasF16)) return false;
  }
  return true;
}

/**
 * Get all available variants for an operation given capabilities.
 * @param {string} operation
 * @param {import('../schema/platform.schema.js').RuntimeCapabilities} capabilities
 * @returns {string[]}
 */
export function getAvailableVariants(operation, capabilities) {
  return getVariantNames(operation).filter(v => isVariantAvailable(operation, v, capabilities));
}

/**
 * Merge base and variant bindings.
 * Variant bindings with matching indices override base bindings.
 * @param {import('../schema/kernel-registry.schema.js').BindingSchema[]} base
 * @param {import('../schema/kernel-registry.schema.js').BindingSchema[] | undefined} override
 * @returns {import('../schema/kernel-registry.schema.js').BindingSchema[]}
 */
export function mergeBindings(base, override) {
  if (!override || override.length === 0) {
    return [...base];
  }

  const result = [...base];
  for (const binding of override) {
    const existingIdx = result.findIndex(b => b.index === binding.index);
    if (existingIdx >= 0) {
      result[existingIdx] = binding;
    } else {
      result.push(binding);
    }
  }

  return result.sort((a, b) => a.index - b.index);
}

/**
 * Resolve a kernel variant to a complete configuration.
 * Merges base operation config with variant-specific overrides.
 * @param {string} operation
 * @param {string} variant
 * @returns {import('../schema/kernel-registry.schema.js').ResolvedKernelConfig | null}
 */
export function resolveKernelConfig(operation, variant) {
  const opSchema = getOperation(operation);
  const variantSchema = getVariant(operation, variant);

  if (!opSchema || !variantSchema) {
    return null;
  }

  return {
    operation,
    variant,
    wgsl: variantSchema.wgsl,
    entryPoint: variantSchema.entryPoint,
    workgroup: variantSchema.workgroup,
    requires: variantSchema.requires ?? [],
    bindings: mergeBindings(opSchema.baseBindings, variantSchema.bindingsOverride),
    uniforms: variantSchema.uniformsOverride ?? opSchema.baseUniforms,
    wgslOverrides: variantSchema.wgslOverrides ?? {},
    sharedMemory: variantSchema.sharedMemory ?? 0,
  };
}
