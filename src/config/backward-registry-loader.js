import backwardRegistryData from './kernels/backward-registry.json' with { type: 'json' };
import { validateBackwardRegistry } from './schema/backward-registry.schema.js';

export function loadBackwardRegistry() {
  return validateBackwardRegistry(backwardRegistryData);
}
