import { bootstrapNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { initDevice, getDevice } from '../../src/gpu/device.js';

const F16_PROBE_SHADER = 'enable f16;\n@compute @workgroup_size(1) fn main() { var x: f16 = 0h; }';

export async function probeNodeGPU({ installFileFetchShim } = {}) {
  try {
    await bootstrapNodeWebGPU();
  } catch {
    return { ready: false, reason: 'bootstrapNodeWebGPU failed' };
  }

  if (typeof globalThis.navigator === 'undefined' || !globalThis.navigator.gpu) {
    return { ready: false, reason: 'navigator.gpu not available' };
  }

  if (typeof globalThis.GPUBuffer === 'undefined') {
    return { ready: false, reason: 'GPUBuffer global not defined' };
  }

  if (installFileFetchShim) {
    try {
      const { installNodeFileFetchShim } = await import('../../src/tooling/node-file-fetch.js');
      installNodeFileFetchShim();
    } catch {
      return { ready: false, reason: 'installNodeFileFetchShim failed' };
    }
  }

  try {
    await initDevice();
  } catch {
    return { ready: false, reason: 'initDevice failed' };
  }

  const device = getDevice();
  if (!device) {
    return { ready: false, reason: 'getDevice returned null' };
  }

  try {
    device.createShaderModule({ code: F16_PROBE_SHADER });
  } catch {
    return { ready: false, reason: 'f16 shader compilation failed' };
  }

  return { ready: true, reason: null };
}
