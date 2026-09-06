import { createDefaultNodeLoadProgressLogger } from './runtime/model-source.js';
import { createDopplerRuntimeService } from './runtime/index.js';
import { createFetchCapsuleArtifactStore } from './runtime/fetch-capsule-artifact-store.js';
import { fetchCapsuleMetadata } from './runtime/capsule-acquisition.js';

async function ensureWebGPUAvailable() {
  if (typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu) {
    return;
  }
  throw new Error('WebGPU is unavailable. Run in a WebGPU-capable browser.');
}

async function resolveCapsuleInput(capsuleSource, options = {}) {
  if (capsuleSource && typeof capsuleSource === 'object') {
    if (!options.artifactStore) {
      throw new Error('doppler.openCapsule(capsuleObject) requires options.artifactStore.');
    }
    return { capsule: capsuleSource, artifactStore: options.artifactStore };
  }
  const capsuleUrl = new URL(capsuleSource, globalThis.location?.href).href;
  return { capsule: await fetchCapsuleMetadata(capsuleUrl, options), artifactStore: options.artifactStore ?? createFetchCapsuleArtifactStore(capsuleUrl) };
}

const runtime = createDopplerRuntimeService({
  ensureWebGPUAvailable,
  defaultLoadProgressLogger: null,
  resolveCapsuleInput,
});

export const doppler = runtime.doppler;
export const load = runtime.load;
export const open = runtime.open;
export const openCapsule = runtime.openCapsule;
export const generate = runtime.generate;
export const clearModelCache = runtime.clearModelCache;
export { createDefaultNodeLoadProgressLogger };

export function resolveLoadProgressHandlers(options = {}) {
  return runtime.resolveLoadProgressHandlers(options);
}

export default doppler;
