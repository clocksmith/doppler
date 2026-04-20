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
