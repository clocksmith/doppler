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
  registry.set(modelType, factory);
}

export function getPipelineFactory(modelType) {
  if (!modelType) return null;
  return registry.get(modelType) ?? null;
}

export function listPipelines() {
  return Array.from(registry.keys());
}
