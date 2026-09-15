import type { KernelConfig } from '../../config/kernel-registry-contract.js';
export declare function getKernelBindGroupLayout(config: KernelConfig, device?: GPUDevice): GPUBindGroupLayout;
export declare function createKernelBindingEntries(config: KernelConfig, resources: Record<string, GPUBufferBinding | null | undefined>): GPUBindGroupEntry[];
