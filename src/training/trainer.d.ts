import type { BackwardRegistrySchema } from '../config/schema/backward-registry.schema.js';
import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { Tensor } from '../gpu/tensor.js';

export interface TrainingBatch {
  input: Tensor;
  targets: Tensor;
}

export interface TrainingOptimizer {
  step: (
    params: Tensor[],
    grads: Map<Tensor, Tensor>,
    config: TrainingConfigSchema
  ) => Promise<void>;
}

export interface TrainStepOptions {
  registry?: BackwardRegistrySchema;
  crossEntropyLoss?: (
    logits: Tensor,
    targets: Tensor,
    config: TrainingConfigSchema,
    tape: unknown
  ) => Promise<Tensor>;
  clipGradients?: (
    grads: Map<Tensor, Tensor>,
    config: TrainingConfigSchema
  ) => Promise<Map<Tensor, Tensor>>;
  optimizer?: TrainingOptimizer;
  lossScale?: number;
  applyClip?: boolean;
  applyOptimizer?: boolean;
}

export interface TrainStepResult {
  loss: Tensor;
  grads: Map<Tensor, Tensor>;
}

export declare function trainStep(
  model: { forward: (input: Tensor, tape: unknown) => Promise<Tensor>; loraParams: () => Tensor[] },
  batch: TrainingBatch,
  config: TrainingConfigSchema,
  options?: TrainStepOptions
): Promise<TrainStepResult>;
