// Narrow GPU-device export for consumers that only need device init and
// capability probing. Tree-shakable away from the mega tooling barrel.

export {
  initDevice,
  getDevice,
  getKernelCapabilities,
  getPlatformConfig,
  isWebGPUAvailable,
} from '../gpu/device.js';
export {
  registerShaderSources,
  hasPreseededShaderSource,
} from '../gpu/kernels/shader-cache.js';
