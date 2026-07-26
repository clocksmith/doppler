/**
 * Kernel runtime initialization helpers.
 */

export function prepareKernelRuntime(
  options?: {
    prewarm?: boolean;
    prewarmMode?: 'parallel' | 'sequential';
    clearCaches?: boolean;
  }
): Promise<{ warmed: boolean }>;
