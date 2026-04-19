/**
 * Kernel runtime initialization helpers.
 */

export function prepareKernelRuntime(
  options?: {
    prewarm?: boolean;
    prewarmMode?: 'parallel' | 'sequential';
    autoTune?: boolean;
    clearCaches?: boolean;
    modelConfig?: Record<string, number>;
  }
): Promise<{ warmed: boolean; tuned: boolean }>;
