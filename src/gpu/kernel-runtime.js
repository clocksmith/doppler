

import { autoTuneKernels, prewarmKernels, clearKernelCaches } from './kernels/utils.js';


export async function prepareKernelRuntime(
  options = {}
) {
  const {
    prewarm = true,
    prewarmMode = 'parallel',
    autoTune = false,
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
