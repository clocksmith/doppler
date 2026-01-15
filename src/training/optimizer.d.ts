import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { Tensor } from '../gpu/tensor.js';

export declare class AdamOptimizer {
  constructor(config: TrainingConfigSchema);
  config: TrainingConfigSchema;
  state: Map<Tensor, { m: Tensor; v: Tensor }>;
  stepCount: number;
  getState(param: Tensor): { m: Tensor; v: Tensor };
  step(params: Tensor[], grads: Map<Tensor, Tensor>, trainingConfig: TrainingConfigSchema): Promise<void>;
}
