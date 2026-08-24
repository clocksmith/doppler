export {
  KERNEL_CONFIGS,
  getKernelConfig,
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
