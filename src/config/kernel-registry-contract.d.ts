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
  validate?: (seqLen: number, numHeads: number, headDim: number) => void;
  outputDtype?: 'f16' | 'f32';
  weightDtype?: string;
  variantMetadata?: VariantMetadata;
}

export const KERNEL_CONFIGS: Record<string, Record<string, KernelConfig>>;
export function getKernelConfig(operation: string, variant: string): KernelConfig;
