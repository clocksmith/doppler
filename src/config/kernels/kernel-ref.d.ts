export interface KernelRefSchema {
  id: string;
  version: string;
  digest: string;
}

export declare const KERNEL_REF_VERSION: string;

export declare function getKernelRefContentDigest(kernel: string, entry?: string): string;
export declare function buildKernelRefFromKernelEntry(kernel: string, entry?: string): KernelRefSchema;
export declare function buildLegacyKernelRefFromKernelEntry(kernel: string, entry?: string): KernelRefSchema;
export declare function isKernelRefBoundToKernel(
  kernelRef: KernelRefSchema | Record<string, unknown> | null | undefined,
  kernel: string,
  entry?: string
): boolean;

