import type { InferenceConfigSchema } from '../config/schema/inference.schema.js';
import type { ManifestInferenceSchema } from '../config/schema/manifest.schema.js';

/**
 * Build RoPE configuration from preset and HF config.
 *
 * HF rope_scaling is treated as source of truth when present.
 */
export declare function buildRoPEConfig(
  presetInference: InferenceConfigSchema,
  config: Record<string, unknown>
): ManifestInferenceSchema['rope'];
