import type { LinearNormMode, ManifestInferenceSchema } from '../../../config/schema/index.js';
import type { Manifest, ParsedModelConfig } from './config.js';

/** Linear-attention fields of ParsedModelConfig. */
export type LinearAttentionConfig = Pick<ParsedModelConfig,
  'linearNumKeyHeads' | 'linearNumValueHeads' | 'linearKeyHeadDim' | 'linearValueHeadDim'
  | 'linearConvKernelDim' | 'linearNormMode' | 'linearNormGateActivation' | 'pairedMechanismReplacements'>;

/**
 * Linear-attention geometry (manifest architecture first, then source config), the shared or
 * per-head norm contract, the declared norm-gate activation, and the paired mechanism programs
 * the manifest's mechanismKernels declare in place of the defaults.
 */
export declare function resolveLinearAttentionConfig(
  arch: Record<string, unknown> & { linearNormMode?: LinearNormMode | string | null },
  config: Record<string, unknown>,
  inf: ManifestInferenceSchema,
  manifest: Manifest,
  modelId: string
): LinearAttentionConfig;
