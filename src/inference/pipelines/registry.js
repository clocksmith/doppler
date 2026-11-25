import { canonicalizeLegacyEnumValue } from '../../config/legacy-enum-values.js';

const registry = new Map();

export function registerPipeline(
  modelType,
  factory
) {
  if (!modelType || typeof modelType !== 'string') {
    throw new Error('Pipeline registry requires a modelType string.');
  }
  if (typeof factory !== 'function') {
    throw new Error(`Pipeline registry requires a factory function for "${modelType}".`);
  }
  registry.set(normalizeModelType(modelType), factory);
}

// Model types are case-insensitive identifiers: lazy module rules already
// match on the lower-cased type, so the registry keys the same way. Legacy
// snake_case spellings of Doppler-owned types resolve to their kebab-case form.
export function normalizeModelType(modelType) {
  return String(canonicalizeLegacyEnumValue(String(modelType).trim().toLowerCase()));
}

export function getPipelineFactory(modelType) {
  if (!modelType || typeof modelType !== 'string') return null;
  return registry.get(normalizeModelType(modelType)) ?? null;
}

export function listPipelines() {
  return Array.from(registry.keys());
}
