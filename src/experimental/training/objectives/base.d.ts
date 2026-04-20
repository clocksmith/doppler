import type { TrainingConfigSchema } from '../../../config/training-defaults.d.ts';
import type { Tensor } from '../../../gpu/tensor.js';

export interface TrainingObjectiveContext {
  model: unknown;
  batch: Record<string, unknown>;
  config: TrainingConfigSchema;
  tape: unknown;
  options: Record<string, unknown>;
  lossScale: number;
}

export interface TrainingObjectiveLossResult {
  loss: Tensor;
  components?: Record<string, number | string | boolean | null | undefined>;
  [key: string]: unknown;
}

export interface TrainingObjectiveBackwardSeed {
  tensor: Tensor;
  grad: Tensor;
}

export type TrainingObjectiveBackwardTargets =
  | Tensor
  | null
  | Map<Tensor, Tensor>
  | TrainingObjectiveBackwardSeed[]
  | { seeds: TrainingObjectiveBackwardSeed[] };

export interface TrainingObjective {
  name: string;
  prepareBatch?: (
    context: Omit<TrainingObjectiveContext, 'batch' | 'tape'> & { batch: Record<string, unknown> }
  ) => Promise<Record<string, unknown>> | Record<string, unknown>;
  forward: (
    context: TrainingObjectiveContext
  ) => Promise<Record<string, unknown>> | Record<string, unknown>;
  computeLoss: (
    context: TrainingObjectiveContext & { forwardState: Record<string, unknown> }
  ) => Promise<Tensor | TrainingObjectiveLossResult> | Tensor | TrainingObjectiveLossResult;
  backwardTargets?: (
    context: TrainingObjectiveContext & { loss: Tensor; lossResult: TrainingObjectiveLossResult }
  ) => Promise<TrainingObjectiveBackwardTargets> | TrainingObjectiveBackwardTargets;
  metrics?: (
    context: TrainingObjectiveContext & {
      forwardState: Record<string, unknown>;
      loss: Tensor;
      lossResult: TrainingObjectiveLossResult;
    }
  ) => Promise<Record<string, number | string | null | undefined>> | Record<string, number | string | null | undefined>;
  cleanup?: (
    context: TrainingObjectiveContext & { preparedBatch: Record<string, unknown> }
  ) => Promise<void> | void;
}

export declare function createTrainingObjective(definition: Partial<TrainingObjective>): TrainingObjective;
export declare function isTrainingObjective(value: unknown): value is TrainingObjective;
