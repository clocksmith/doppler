export {
  KERNEL_CONFIGS,
} from '../../config/kernel-registry-contract.js';
export type {
  KernelConfig,
  VariantMetadata,
} from '../../config/kernel-registry-contract.js';

export function setKernelValidator(
  operation: string,
  variant: string,
  validator: (seqLen: number, numHeads: number, headDim: number) => void
): void;

import type { KernelRegistry, KernelRegistryOptions, KernelConfig, KernelValidator } from '../../config/kernel-registry-contract.js';
export declare const DEFAULT_KERNEL_REGISTRY: KernelRegistry;
export declare function createKernelRegistry(options?: KernelRegistryOptions): KernelRegistry;
export declare function getKernelRegistry(): KernelRegistry;
export declare function getKernelConfig(operation: string, variant: string): KernelConfig;
export declare function getKernelValidator(operation: string, variant: string): KernelValidator['validate'] | null;
/** Internal compatibility lease; caller must serialize the entire operation. */
export declare function enterKernelRegistry(registry: KernelRegistry): () => void;
