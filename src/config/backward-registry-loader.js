const loadJson = async (path) => {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw new Error(`Failed to load json: ${path}`);
  return response.json();
};

const backwardRegistryData = await loadJson('./kernels/backward-registry.json');
import { validateBackwardRegistry } from './schema/backward-registry.schema.js';

export function loadBackwardRegistry() {
  return validateBackwardRegistry(backwardRegistryData);
}
