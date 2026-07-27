export declare const KERNEL_NEGATIVE_RESULTS_SCHEMA:
  'doppler.kernel-negative-results/v1';

export declare function validateKernelNegativeResults(
  input: Record<string, unknown>
): Record<string, unknown>;

export declare function findKernelNegativeResults(
  input: Record<string, unknown>,
  scope?: {
    modelId?: string;
    candidate?: string;
    adapterDigest?: string;
    phase?: string;
  }
): Array<Record<string, unknown>>;
