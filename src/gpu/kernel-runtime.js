

import { autoTuneKernels, prewarmKernels, clearKernelCaches } from './kernels/utils.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { DEFAULT_KERNEL_WARMUP_CONFIG } from '../config/schema/kernel-warmup.schema.js';


export async function prepareKernelRuntime(
  options = {}
) {
  const kernelWarmup = getRuntimeConfig().shared?.kernelWarmup ?? DEFAULT_KERNEL_WARMUP_CONFIG;
  const {
    prewarm = kernelWarmup.prewarm,
    prewarmMode = kernelWarmup.prewarmMode,
    autoTune = kernelWarmup.autoTune,
    clearCaches = false,
    modelConfig = {},
  } = options;

  if (clearCaches) {
    clearKernelCaches();
  }

  let tuned = false;
  if (autoTune) {
    await autoTuneKernels(modelConfig);
    tuned = true;
  }

  let warmed = false;
  if (prewarm) {
    await prewarmKernels({ mode: prewarmMode });
    warmed = true;
  }

  return { warmed, tuned };
}
