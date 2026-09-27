export interface VariantMetadata {
  colsPerWg?: number;
  tileM?: number;
  outputBinding?: number;
  maxKVLen?: number;
  [key: string]: unknown;
}

import type { BindingSchema, UniformsSchema } from './schema/kernel-registry.schema.js';

export interface KernelConfig {
  operation: string;
  variant: string;
  shaderFile: string;
  entryPoint: string;
  workgroupSize: [number, number, number];
  requires: string[];
  requiredWgslFeatures: string[];
  bindings: BindingSchema[];
  readonly uniforms: UniformsSchema | null;
  wgslOverrides?: Record<string, unknown>;
  sharedMemory?: number;
  outputDtype?: 'f16' | 'f32';
  weightDtype?: string;
  variantMetadata?: VariantMetadata;
}

export const KERNEL_CONFIGS: Record<string, Record<string, KernelConfig>>;
export function getKernelConfig(operation: string, variant: string): KernelConfig;

export interface KernelValidator {
  readonly id: string;
  readonly validate: (seqLen: number, numHeads: number, headDim: number) => void;
}
export interface KernelRegistry {
  readonly identity: string;
  readonly configs: Readonly<Record<string, Readonly<Record<string, KernelConfig>>>>;
  readonly validators: Readonly<Record<string, Readonly<Record<string, KernelValidator>>>>;
  getKernelConfig(operation: string, variant: string): KernelConfig;
  getKernelValidator(operation: string, variant: string): KernelValidator['validate'] | null;
}
export interface KernelRegistryOptions {
  extensions?: Record<string, Partial<import('./schema/kernel-registry.schema.js').OperationSchema>>;
  validators?: Record<string, Record<string, KernelValidator>>;
}
export declare function createKernelRegistry(options?: KernelRegistryOptions): KernelRegistry;
export declare const DEFAULT_KERNEL_REGISTRY: KernelRegistry;
export declare function isKernelRegistry(value: unknown): value is KernelRegistry;

export declare function getActiveKernelRegistry(): KernelRegistry | null;
export declare function enterKernelRegistry(registry: KernelRegistry): () => void;
export declare function getKernelConfigs(): KernelRegistry['configs'];
