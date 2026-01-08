import { getDevice, setDevice } from '../../gpu/device.js';
import { applyDebugConfig, setGPUDevice } from '../../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';

export function applyPipelineContexts(target, contexts = {}, options = {}) {
  const runtimeConfig = contexts.runtimeConfig
    ? setRuntimeConfig(contexts.runtimeConfig)
    : getRuntimeConfig();
  const sharedDebug = runtimeConfig.shared?.debug;

  if (options.applySharedDebug !== false && sharedDebug) {
    applyDebugConfig(sharedDebug);
  }

  if (contexts.gpu?.device) {
    const device = contexts.gpu.device;
    setDevice(device);
    setGPUDevice(device);
    if (options.assignGpuContext) {
      target.gpuContext = { device };
    }
    if (options.assignUseGPU) {
      target.useGPU = true;
    }
  } else {
    const device = getDevice();
    if (device) setGPUDevice(device);
  }

  if (options.assignMemoryContext && contexts.memory) {
    target.memoryContext = contexts.memory;
  }
  if (options.assignStorageContext && contexts.storage) {
    target.storageContext = contexts.storage;
  }
  if (options.assignBaseUrl !== false && contexts.baseUrl) {
    target.baseUrl = contexts.baseUrl;
  }
  if (options.assignProgress !== false && contexts.onProgress) {
    target._onProgress = contexts.onProgress;
  }

  return { runtimeConfig, sharedDebug };
}
