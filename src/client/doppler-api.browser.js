import { createDefaultNodeLoadProgressLogger } from './runtime/model-source.js';
import { createDopplerRuntimeService } from './runtime/index.js';
import { createFetchPackArtifactStore } from './runtime/fetch-pack-artifact-store.js';

async function ensureWebGPUAvailable() {
  if (typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu) {
    return;
  }
  throw new Error('WebGPU is unavailable. Run in a WebGPU-capable browser.');
}

async function resolvePackInput(packSource, options = {}) {
  if (packSource && typeof packSource === 'object') {
    if (!options.artifactStore) {
      throw new Error('doppler.openPack(packObject) requires options.artifactStore.');
    }
    return { pack: packSource, artifactStore: options.artifactStore };
  }
  const packUrl = new URL(packSource, globalThis.location?.href).href;
  const response = await fetch(packUrl);
  if (!response.ok) throw new Error(`Doppler Pack fetch failed (${response.status}) for ${packUrl}.`);
  return { pack: await response.json(), artifactStore: createFetchPackArtifactStore(packUrl) };
}

const runtime = createDopplerRuntimeService({
  ensureWebGPUAvailable,
  defaultLoadProgressLogger: null,
  resolvePackInput,
});

export const doppler = runtime.doppler;
export const load = runtime.load;
export const open = runtime.open;
export const openPack = runtime.openPack;
export const generate = runtime.generate;
export const clearModelCache = runtime.clearModelCache;
export { createDefaultNodeLoadProgressLogger };

export function resolveLoadProgressHandlers(options = {}) {
  return runtime.resolveLoadProgressHandlers(options);
}

export default doppler;
