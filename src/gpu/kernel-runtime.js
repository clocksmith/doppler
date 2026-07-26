

import { prewarmKernels, clearKernelCaches } from './kernels/utils.js';
import { getRuntimeConfig } from '../config/runtime.js';


export async function prepareKernelRuntime(
  options = {}
) {
  const kernelWarmup = getRuntimeConfig().shared?.kernelWarmup;
  if (!kernelWarmup) {
    throw new Error('runtime.shared.kernelWarmup is required but missing from resolved config');
  }
  const {
    prewarm = kernelWarmup.prewarm,
    prewarmMode = kernelWarmup.prewarmMode,
    clearCaches = false,
  } = options;

  if (clearCaches) {
    clearKernelCaches();
  }

  let warmed = false;
  if (prewarm) {
    await prewarmKernels({ mode: prewarmMode });
    warmed = true;
  }

  return { warmed };
}
