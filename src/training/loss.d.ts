import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { Tensor } from '../gpu/tensor.js';
import type { AutogradTape } from './autograd.js';

export declare function crossEntropyLoss(
  logits: Tensor,
  targets: Tensor,
  config: TrainingConfigSchema,
  tape: AutogradTape
): Promise<Tensor>;
