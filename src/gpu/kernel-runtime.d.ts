/**
 * Kernel runtime initialization helpers.
 */

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

export function prepareKernelRuntime(
  options?: KernelRuntimeOptions
): Promise<KernelRuntimeState>;
