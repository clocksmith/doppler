function asOptionalFunction(value, label) {
  if (value == null) return null;
  if (typeof value !== 'function') {
    throw new Error(`Training objective: ${label} must be a function when provided.`);
  }
  return value;
}

export function createTrainingObjective(definition = {}) {
  if (!definition || typeof definition !== 'object' || Array.isArray(definition)) {
    throw new Error('Training objective must be an object.');
  }
  const name = typeof definition.name === 'string' && definition.name.trim()
    ? definition.name.trim()
    : 'objective';
  const forward = asOptionalFunction(definition.forward, 'forward');
  const computeLoss = asOptionalFunction(definition.computeLoss, 'computeLoss');
  if (!forward || !computeLoss) {
    throw new Error('Training objective requires forward() and computeLoss().');
  }

  return Object.freeze({
    name,
    prepareBatch: asOptionalFunction(definition.prepareBatch, 'prepareBatch'),
    forward,
    computeLoss,
    backwardTargets: asOptionalFunction(definition.backwardTargets, 'backwardTargets'),
    metrics: asOptionalFunction(definition.metrics, 'metrics'),
    cleanup: asOptionalFunction(definition.cleanup, 'cleanup'),
  });
}

export function isTrainingObjective(value) {
  return !!value
    && typeof value === 'object'
    && typeof value.forward === 'function'
    && typeof value.computeLoss === 'function';
}
