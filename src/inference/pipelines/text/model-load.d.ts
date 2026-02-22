import type {
  RuntimeConfigSchema,
  KernelWarmupConfigSchema,
  KernelPathSchema,
} from '../../../config/schema/index.js';
import type { KernelPathSource } from '../../../config/kernel-path-loader.js';
import type { ParsedModelConfig, Manifest } from './config.js';
import type { Tokenizer } from '../../tokenizer.js';

export interface KernelWarmupOptions {
  useGPU: boolean;
  kernelWarmup?: KernelWarmupConfigSchema | null;
  modelConfig: ParsedModelConfig;
}

export interface KernelPathResolutionOptions {
  manifest: Manifest;
  runtimeConfig: RuntimeConfigSchema;
  modelConfig: ParsedModelConfig;
}

export interface KernelPathResolutionResult {
  resolvedKernelPath: KernelPathSchema | null;
  kernelPathSource: KernelPathSource;
  runtimeConfig: RuntimeConfigSchema;
}

export function runKernelWarmup(options: KernelWarmupOptions): Promise<void>;

export function resolveAndActivateKernelPath(
  options: KernelPathResolutionOptions
): KernelPathResolutionResult;

export function initTokenizerFromManifestPreset(
  manifest: Manifest,
  baseUrl?: string | null
): Promise<Tokenizer>;
