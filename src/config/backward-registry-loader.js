import { loadJson } from '../utils/load-json.js';
import { validateBackwardRegistry } from './schema/backward-registry.schema.js';

const backwardRegistryData = await loadJson('./kernels/backward-registry.json', import.meta.url, 'Failed to load json');

export function loadBackwardRegistry() {
  return validateBackwardRegistry(backwardRegistryData);
}
