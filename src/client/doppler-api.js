import { createDefaultNodeLoadProgressLogger } from './runtime/model-source.js';
import { createDopplerRuntimeService } from './runtime/index.js';

function isNodeRuntime() {
  return typeof process !== 'undefined'
    && typeof process.versions === 'object'
    && typeof process.versions.node === 'string';
}

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

const runtime = createDopplerRuntimeService({
  ensureWebGPUAvailable,
  defaultLoadProgressLogger: createDefaultNodeLoadProgressLogger(),
});

export const doppler = runtime.doppler;
export const load = runtime.load;
export const clearModelCache = runtime.clearModelCache;
export { createDefaultNodeLoadProgressLogger };

export function resolveLoadProgressHandlers(options = {}) {
  return runtime.resolveLoadProgressHandlers(options);
}

export default doppler;
