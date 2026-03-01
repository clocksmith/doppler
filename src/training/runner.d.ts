import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { Tensor } from '../gpu/tensor.js';
import type { DynamicLossScaler } from './loss-scaling.js';
import type { TrainingBatch, TrainingOptimizer, ClipMetrics } from './trainer.js';
import type { DataLoader } from './dataloader.js';
import type { TrainingObjective } from './objectives/base.js';
import type { UlArtifactFinalizeResult } from './artifacts.js';

export interface TrainingStepMetricsEntry {
  schemaVersion: number;
  step: number;
  epoch: number;
  batch: number;
  total_loss: number;
  step_time_ms: number;
  forward_ms?: number;
  backward_ms?: number;
  optimizer_ms?: number;
  effective_lr?: number | null;
  scheduler_index?: number | null;
  scheduler_phase?: string | null;
  gradient_norm_unclipped?: number;
  gradient_norm_clipped?: number;
  clipped_event_count?: number;
  total_param_count?: number;
  trainable_param_count?: number | null;
  trainable_groups?: string[];
  frozen_groups?: string[];
  nan_count?: number;
  inf_count?: number;
  saturation_count?: number;
  telemetry_mode?: 'step' | 'window' | 'epoch';
  telemetry_window_size?: number;
  telemetry_alerts?: string[];
  window_loss_avg?: number | null;
  window_step_time_ms_avg?: number | null;
  ul_stage?: string | null;
  lambda?: number | null;
  objective?: string;
  loss_total?: number | null;
  coeff_ce?: number | null;
  coeff_prior?: number | null;
  coeff_decoder?: number | null;
  coeff_recon?: number | null;
  schedule_step_index?: number | null;
  latent_clean_mean?: number | null;
  latent_clean_std?: number | null;
  latent_noise_mean?: number | null;
  latent_noise_std?: number | null;
  latent_noisy_mean?: number | null;
  latent_noisy_std?: number | null;
  stage1_latent_count?: number | null;
  loss_prior?: number | null;
  loss_decoder?: number | null;
  loss_recon?: number | null;
  latent_bitrate_proxy?: number | null;
  [key: string]: unknown;
}

export interface TrainingRunnerCallbacks {
  onStep?: (entry: TrainingStepMetricsEntry) => Promise<void> | void;
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
  ) => Promise<ClipMetrics>;
  lossScaler?: DynamicLossScaler;
  trainingObjective?: TrainingObjective;
}

export interface TrainingRunOptions {
  epochs?: number;
  batchSize?: number;
  shuffle?: boolean;
  maxSteps?: number | null;
  logEvery?: number;
  prepareBatch?: (batch: unknown) => Promise<TrainingBatch> | TrainingBatch;
  ulArtifactDir?: string | null;
  modelId?: string | null;
  modelUrl?: string | null;
  timestamp?: string | Date | null;
  persistStage1Latents?: boolean;
}

export declare class TrainingRunner {
  constructor(config: TrainingConfigSchema, options?: TrainingRunnerOptions);
  lastArtifact: UlArtifactFinalizeResult | null;
  run(
    model: {
      forward: (input: Tensor, tape: unknown) => Promise<Tensor>;
      loraParams?: () => Tensor[];
      paramGroups?: () => Record<string, Tensor[]>;
    },
    dataset: TrainingBatch[] | DataLoader<TrainingBatch> | unknown[],
    options?: TrainingRunOptions
  ): Promise<TrainingStepMetricsEntry[]>;
}

export declare function runTraining(
  model: {
    forward: (input: Tensor, tape: unknown) => Promise<Tensor>;
    loraParams?: () => Tensor[];
    paramGroups?: () => Record<string, Tensor[]>;
  },
  dataset: TrainingBatch[] | DataLoader<TrainingBatch> | unknown[],
  config: TrainingConfigSchema,
  options?: TrainingRunOptions & TrainingRunnerOptions
): Promise<TrainingStepMetricsEntry[]>;
