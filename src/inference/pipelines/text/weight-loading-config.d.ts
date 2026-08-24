import type { KernelPathSchema } from '../../../config/schema/index.js';
import type { KernelPathSource } from '../../../config/kernel-path-loader.js';
import type { Manifest } from './config.js';

export interface ResolvedWeightLoadingConfig {
  useFusedQ4K: boolean;
  q4kLayout: 'row' | 'col' | null;
  keepF32Weights: boolean;
  keepBF16Weights: boolean;
  q4kMaterializationMode: 'dense' | 'fused' | 'mixed';
  q4kFusedRoles: string[];
}

export type ResolvedQ4KConfig = ResolvedWeightLoadingConfig;

export function resolveWeightLoadingConfig(
  manifest: Manifest,
  kernelPath?: KernelPathSchema | null,
  kernelPathSource?: KernelPathSource,
  keepF32Weights?: boolean
): ResolvedWeightLoadingConfig;

export const resolveQ4KConfig: typeof resolveWeightLoadingConfig;
