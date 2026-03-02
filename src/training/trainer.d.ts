import type { BackwardRegistrySchema } from '../config/schema/backward-registry.schema.js';
import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { Tensor } from '../gpu/tensor.js';
import type { TrainingObjective } from './objectives/base.js';

export interface TrainingBatch {
  input: Tensor;
  targets: Tensor;
}

export interface TrainingOptimizer {
  step: (
    params: Tensor[],
    grads: Map<Tensor, Tensor>,
    config: TrainingConfigSchema,
    context?: {
      trainableGroups?: string[];
      frozenGroups?: string[];
      allGroups?: string[];
    }
  ) => Promise<OptimizerMetrics>;
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
  ) => Promise<ClipMetrics>;
  optimizer?: TrainingOptimizer;
  lossScale?: number;
  applyClip?: boolean;
  applyOptimizer?: boolean;
  trainingObjective?: TrainingObjective;
  stepIndex?: number | null;
  epochIndex?: number | null;
  batchIndex?: number | null;
  stage1ArtifactContext?: Record<string, unknown> | null;
  stageAArtifactContext?: Record<string, unknown> | null;
}

export interface ClipMetrics {
  clippedGrads: Map<Tensor, Tensor>;
  gradient_norm_unclipped: number;
  gradient_norm_clipped: number;
  clipped_event_count: number;
  total_param_count: number;
}

export interface OptimizerMetrics {
  optimizer_ms: number;
  effective_lr?: number | null;
  scheduler_index?: number | null;
  scheduler_phase?: string | null;
}

export interface TrainStepResult {
  loss: Tensor;
  grads: Map<Tensor, Tensor>;
  forward_ms?: number;
  backward_ms?: number;
  clipMetrics?: ClipMetrics;
  optimizerMetrics?: OptimizerMetrics;
  objectiveName?: string;
  objectiveMetrics?: Record<string, number | string | boolean | null | undefined>;
  paramGroupMetrics?: {
    trainableGroups: string[];
    frozenGroups: string[];
    allGroups: string[];
    trainableParamCount: number;
  };
}

export declare function trainStep(
  model: {
    forward: (input: Tensor, tape: unknown) => Promise<Tensor>;
    loraParams?: () => Tensor[];
    paramGroups?: () => Record<string, Tensor[]>;
  },
  batch: TrainingBatch,
  config: TrainingConfigSchema,
  options?: TrainStepOptions
): Promise<TrainStepResult>;
