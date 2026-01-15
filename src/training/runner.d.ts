import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { Tensor } from '../gpu/tensor.js';
import type { DynamicLossScaler } from './loss-scaling.js';
import type { TrainingBatch, TrainingOptimizer } from './trainer.js';
import type { DataLoader } from './dataloader.js';

export interface TrainingRunnerCallbacks {
  onStep?: (entry: { step: number; epoch: number; batch: number; loss: number }) => Promise<void> | void;
  onEpoch?: (entry: { epoch: number; steps: number; loss: number }) => Promise<void> | void;
}

export interface TrainingRunnerOptions extends TrainingRunnerCallbacks {
  optimizer?: TrainingOptimizer;
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
  lossScaler?: DynamicLossScaler;
}

export interface TrainingRunOptions {
  epochs?: number;
  batchSize?: number;
  shuffle?: boolean;
  maxSteps?: number | null;
  logEvery?: number;
  prepareBatch?: (batch: unknown) => Promise<TrainingBatch> | TrainingBatch;
}

export declare class TrainingRunner {
  constructor(config: TrainingConfigSchema, options?: TrainingRunnerOptions);
  run(
    model: { forward: (input: Tensor, tape: unknown) => Promise<Tensor>; loraParams: () => Tensor[] },
    dataset: TrainingBatch[] | DataLoader<TrainingBatch> | unknown[],
    options?: TrainingRunOptions
  ): Promise<Array<{ step: number; epoch: number; batch: number; loss: number }>>;
}

export declare function runTraining(
  model: { forward: (input: Tensor, tape: unknown) => Promise<Tensor>; loraParams: () => Tensor[] },
  dataset: TrainingBatch[] | DataLoader<TrainingBatch> | unknown[],
  config: TrainingConfigSchema,
  options?: TrainingRunOptions & TrainingRunnerOptions
): Promise<Array<{ step: number; epoch: number; batch: number; loss: number }>>;
