import { loadJson } from '../utils/load-json.js';
import { validateBackwardRegistry } from './schema/backward-registry.schema.js';

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

const backwardRegistryData = deepFreeze(
  validateBackwardRegistry(
    await loadJson('./kernels/backward-registry.json', import.meta.url, 'Failed to load json')
  )
);

export function loadBackwardRegistry() {
  return backwardRegistryData;
}
