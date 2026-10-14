export interface VariantMetadata {
  pairedMechanismKernels?: readonly { operation: string; variant: string }[];
  mechanismKernels?: readonly { operation: string; variant: string }[];
  colsPerWg?: number;
  tileM?: number;
  outputBinding?: number;
  maxKVLen?: number;
  /** Gate activation applied by a gated-delta kernel's gated RMSNorm; guarded against the manifest. */
  normGateActivation?: 'silu' | 'sigmoid';
  [key: string]: unknown;
}

import type { BindingSchema, UniformsSchema } from './schema/kernel-registry.schema.js';

export type DeepReadonly<T> = T extends (...args: never[]) => unknown ? T
  : T extends readonly unknown[] ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T extends object ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T;

export interface KernelConfig {
  readonly operation: string;
  readonly variant: string;
  readonly shaderFile: string;
  readonly entryPoint: string;
  readonly immediateCompletion: 'submitted' | 'settled';
  readonly workgroupSize: readonly [number, number, number];
  readonly requires: readonly string[];
  readonly requiredWgslFeatures: readonly string[];
  readonly bindings: readonly DeepReadonly<BindingSchema>[];
  readonly uniforms: DeepReadonly<UniformsSchema> | null;
  readonly wgslOverrides?: DeepReadonly<Record<string, unknown>>;
  readonly sharedMemory?: number;
  /** Raw shader output; composed wrappers may apply declared subsequent stages. */
  readonly outputDtype?: 'f16' | 'f32' | 'i32';
  readonly weightDtype?: string;
  readonly variantMetadata?: DeepReadonly<VariantMetadata>;
}

export const KERNEL_CONFIGS: Readonly<Record<string, Readonly<Record<string, KernelConfig>>>>;
export function getKernelConfig(operation: string, variant: string): KernelConfig;

export interface KernelValidationContext {
  readonly operation: string;
  readonly variant: string;
  readonly bindings: readonly unknown[];
  readonly uniforms: Readonly<Record<string, number>>;
  readonly workgroups: number | readonly number[] | Readonly<{ indirectBuffer: GPUBuffer; indirectOffset?: number }>;
  readonly constants: Readonly<Record<string, number | boolean>> | null;
  readonly extraBindings: readonly unknown[] | null;
}

export interface KernelValidator {
  readonly id: string;
  readonly validate: (context: KernelValidationContext) => void;
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

export declare function getKernelRegistryIdentity(): string;
