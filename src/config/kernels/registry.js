let cachedRegistry = null;

let registryUrl = null;

export function setRegistryUrl(url) {
  registryUrl = url;
  cachedRegistry = null;
}

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

export function getRegistrySync() {
  if (!cachedRegistry) {
    throw new Error('Kernel registry not loaded. Call await getRegistry() first.');
  }
  return cachedRegistry;
}

export function clearRegistryCache() {
  cachedRegistry = null;
}

export function getOperation(operation) {
  const registry = getRegistrySync();
  return registry.operations[operation];
}

export function getVariant(operation, variant) {
  const op = getOperation(operation);
  return op?.variants[variant];
}

export function getVariantNames(operation) {
  const op = getOperation(operation);
  return op ? Object.keys(op.variants) : [];
}

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

export function getAvailableVariants(operation, capabilities) {
  return getVariantNames(operation).filter(v => isVariantAvailable(operation, v, capabilities));
}

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
