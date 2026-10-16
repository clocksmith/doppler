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

export function resolveKernelConfig(operation, variant) {
  const opSchema = getOperation(operation);
  const variantSchema = getVariant(operation, variant);

  if (!opSchema || !variantSchema) {
    return null;
  }

  return resolveSchema(operation, variant, opSchema, variantSchema);
}
