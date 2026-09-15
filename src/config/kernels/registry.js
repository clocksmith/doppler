import { resolveKernelConfig as resolveSchema } from '../schema/kernel-registry.schema.js';
export { mergeBindings } from '../schema/kernel-registry.schema.js';
import { loadJson } from '../../formats/load-json.js';

let cachedRegistry = null;

let registryUrl = null;

function deepFreeze(value, seen = new WeakSet()) {
  if (!value || typeof value !== 'object' || seen.has(value)) {
    return value;
  }
  seen.add(value);
  for (const entry of Object.values(value)) {
    deepFreeze(entry, seen);
  }
  return Object.freeze(value);
}

export function setRegistryUrl(url) {
  registryUrl = url;
  cachedRegistry = null;
}

export async function getRegistry() {
  if (cachedRegistry) {
    return cachedRegistry;
  }

  const source = registryUrl || './registry.json';
  cachedRegistry = deepFreeze(
    await loadJson(source, import.meta.url, 'Failed to load kernel registry')
  );
  return cachedRegistry;
}

function getRegistrySync() {
  if (!cachedRegistry) {
    throw new Error('Kernel registry not loaded. Call await getRegistry() first.');
  }
  return cachedRegistry;
}

export function clearRegistryCache() {
  cachedRegistry = null;
}

function getOperation(operation) {
  const registry = getRegistrySync();
  return registry.operations[operation];
}

function getVariant(operation, variant) {
  const op = getOperation(operation);
  return op?.variants[variant];
}

function getVariantNames(operation) {
  const op = getOperation(operation);
  return op ? Object.keys(op.variants) : [];
}

function isVariantAvailable(operation, variant, capabilities) {
  const variantSchema = getVariant(operation, variant);
  if (!variantSchema) return false;

  const requires = variantSchema.requires || [];
  for (const req of requires) {
    if (req === 'shader-f16' && !capabilities.hasF16) return false;
    if (req === 'subgroups' && !capabilities.hasSubgroups) return false;
  }
  return true;
}

export function resolveKernelConfig(operation, variant) {
  const opSchema = getOperation(operation);
  const variantSchema = getVariant(operation, variant);

  if (!opSchema || !variantSchema) {
    return null;
  }

  return resolveSchema(operation, variant, opSchema, variantSchema);
}
