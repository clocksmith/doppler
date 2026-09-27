import type { KernelRegistry } from '../../config/kernel-registry-contract.js';
export declare function listPrewarmKernels(
  registry: KernelRegistry,
  capabilities: { hasF16: boolean; hasSubgroups: boolean; wgslLanguageFeatures?: readonly string[] }
): Array<[string, Array<[string, KernelRegistry['configs'][string][string]]>]>;
export declare function prewarmKernels(options?: {
  mode?: 'parallel' | 'sequential';
}): Promise<void>;
