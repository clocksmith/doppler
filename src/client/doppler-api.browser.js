import { createDefaultNodeLoadProgressLogger } from './runtime/model-source.js';
import { createDopplerRuntimeService } from './runtime/index.js';

async function ensureWebGPUAvailable() {
  if (typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu) {
    return;
  }
  throw new Error('WebGPU is unavailable. Run in a WebGPU-capable browser.');
}

const runtime = createDopplerRuntimeService({
  ensureWebGPUAvailable,
  defaultLoadProgressLogger: null,
});

export const doppler = runtime.doppler;
export const load = runtime.load;
export const clearModelCache = runtime.clearModelCache;
export { createDefaultNodeLoadProgressLogger };

export function resolveLoadProgressHandlers(options = {}) {
  return runtime.resolveLoadProgressHandlers(options);
}

export default doppler;
