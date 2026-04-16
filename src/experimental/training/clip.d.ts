import type { TrainingConfigSchema } from '../../config/training-defaults.d.ts';
import type { Tensor } from '../../gpu/tensor.js';

import type { ClipMetrics } from './trainer.d.ts';

export declare function clipGradients(
  grads: Map<Tensor, Tensor>,
  config: TrainingConfigSchema
): Promise<ClipMetrics>;
