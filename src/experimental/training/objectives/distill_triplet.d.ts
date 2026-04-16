import type { Tensor } from '../../gpu/tensor.js';
import type { TrainingConfigSchema } from '../../config/training-defaults.d.ts';
import type { TrainingObjective } from './base.js';

export interface DistillTripletObjectiveOptions {
  crossEntropyLoss?: (
    logits: Tensor,
    targets: Tensor,
    config: TrainingConfigSchema,
    tape: unknown
  ) => Promise<Tensor>;
}

export declare function createDistillTripletObjective(
  options?: DistillTripletObjectiveOptions
): TrainingObjective;
