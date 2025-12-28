/**
 * Kernel runtime initialization helpers.
 */

import { autoTuneKernels, prewarmKernels, clearKernelCaches } from './kernels/utils.js';

export interface KernelRuntimeOptions {
  prewarm?: boolean;
  prewarmMode?: 'parallel' | 'sequential';
  autoTune?: boolean;
  clearCaches?: boolean;
  modelConfig?: Record<string, number>;
}

export interface KernelRuntimeState {
  warmed: boolean;
  tuned: boolean;
}

export async function prepareKernelRuntime(
  options: KernelRuntimeOptions = {}
): Promise<KernelRuntimeState> {
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
