import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { Tensor } from '../gpu/tensor.js';

export declare function clipGradients(
  grads: Map<Tensor, Tensor>,
  config: TrainingConfigSchema
): Promise<Map<Tensor, Tensor>>;
