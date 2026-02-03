/**
 * SD3 Transformer GPU path.
 *
 * @module inference/diffusion/sd3-transformer
 */

import type { Tensor } from '../../gpu/tensor.js';
import type { DiffusionModelConfig, DiffusionRuntimeConfig } from './types.js';
import type { DiffusionWeightEntry } from './weights.js';

export declare function runSD3Transformer(
  latents: Tensor,
  context: Tensor,
  timeText: Tensor,
  weightsEntry: DiffusionWeightEntry,
  modelConfig: DiffusionModelConfig,
  runtime: DiffusionRuntimeConfig
): Promise<Tensor>;
