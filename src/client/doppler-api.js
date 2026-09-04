import { createDefaultNodeLoadProgressLogger } from './runtime/model-source.js';
import { createDopplerRuntimeService } from './runtime/index.js';
import { isNodeRuntime } from '../storage/runtime-env.js';
import { createFetchPackArtifactStore } from './runtime/fetch-pack-artifact-store.js';

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

async function resolvePackInput(packSource, options = {}) {
  if (packSource && typeof packSource === 'object') {
    if (!options.artifactStore) {
      throw new Error('doppler.openPack(packObject) requires options.artifactStore.');
    }
    return { pack: packSource, artifactStore: options.artifactStore };
  }
  if (typeof packSource !== 'string' || !packSource.trim()) {
    throw new Error('doppler.openPack() requires a Pack object, path, or URL.');
  }
  let parsedUrl = null;
  try {
    parsedUrl = new URL(packSource);
  } catch {
    parsedUrl = null;
  }
  if (parsedUrl && parsedUrl.protocol !== 'file:') {
    const response = await fetch(parsedUrl.href);
    if (!response.ok) throw new Error(`Doppler Pack fetch failed (${response.status}) for ${parsedUrl.href}.`);
    return { pack: await response.json(), artifactStore: options.artifactStore ?? createFetchPackArtifactStore(parsedUrl.href) };
  }
  if (!isNodeRuntime()) throw new Error('Browser doppler.openPack() requires an HTTP(S) Pack URL.');
  const [{ fileURLToPath }, pathModule, { loadPack }, { createNodePackArtifactStore }] = await Promise.all([
    import('node:url'),
    import('node:path'),
    import('../tooling/pack.js'),
    import('../tooling/node-pack-artifact-store.js'),
  ]);
  const packPath = parsedUrl?.protocol === 'file:'
    ? fileURLToPath(parsedUrl)
    : pathModule.resolve(packSource);
  return {
    pack: await loadPack(packPath),
    artifactStore: options.artifactStore ?? createNodePackArtifactStore(packPath),
  };
}

const runtime = createDopplerRuntimeService({
  ensureWebGPUAvailable,
  defaultLoadProgressLogger: createDefaultNodeLoadProgressLogger(),
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
