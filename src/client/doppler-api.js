import { createDefaultNodeLoadProgressLogger } from './runtime/model-source.js';
import { createDopplerRuntimeService } from './runtime/index.js';
import { isNodeRuntime } from '../storage/runtime-env.js';
import { createFetchCapsuleArtifactStore } from './runtime/fetch-capsule-artifact-store.js';
import { fetchCapsuleMetadata } from './runtime/capsule-acquisition.js';

async function ensureWebGPUAvailable() {
  if (typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu) {
    return;
  }
  if (isNodeRuntime()) {
    const { bootstrapNodeWebGPU } = await import('../tooling/node-webgpu.js');
    const result = await bootstrapNodeWebGPU();
    if (result.ok && globalThis.navigator?.gpu) {
      return;
    }
  }
  throw new Error('WebGPU is unavailable. Install a Node WebGPU provider or run in a WebGPU-capable browser.');
}

async function resolveCapsuleInput(capsuleSource, options = {}) {
  if (capsuleSource && typeof capsuleSource === 'object') {
    if (!options.artifactStore) {
      throw new Error('doppler.openCapsule(capsuleObject) requires options.artifactStore.');
    }
    return { capsule: capsuleSource, artifactStore: options.artifactStore };
  }
  if (typeof capsuleSource !== 'string' || !capsuleSource.trim()) {
    throw new Error('doppler.openCapsule() requires a Capsule object, path, or URL.');
  }
  let parsedUrl = null;
  try {
    parsedUrl = new URL(capsuleSource);
  } catch {
    parsedUrl = null;
  }
  if (parsedUrl && parsedUrl.protocol !== 'file:') {
    return { capsule: await fetchCapsuleMetadata(parsedUrl.href, options), artifactStore: options.artifactStore ?? createFetchCapsuleArtifactStore(parsedUrl.href) };
  }
  if (!isNodeRuntime()) throw new Error('Browser doppler.openCapsule() requires an HTTP(S) Capsule URL.');
  const [{ fileURLToPath }, pathModule, { loadCapsule }, { createNodeCapsuleArtifactStore }] = await Promise.all([
    import('node:url'),
    import('node:path'),
    import('../tooling/capsule.js'),
    import('../tooling/node-capsule-artifact-store.js'),
  ]);
  const capsulePath = parsedUrl?.protocol === 'file:'
    ? fileURLToPath(parsedUrl)
    : pathModule.resolve(capsuleSource);
  return {
    capsule: await loadCapsule(capsulePath, { signal: options.signal }),
    artifactStore: options.artifactStore ?? createNodeCapsuleArtifactStore(capsulePath),
  };
}

const runtime = createDopplerRuntimeService({
  ensureWebGPUAvailable,
  defaultLoadProgressLogger: createDefaultNodeLoadProgressLogger(),
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
