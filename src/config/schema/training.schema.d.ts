import type { LoraConfigSchema } from './lora.schema.js';
import type { UlTrainingConfigSchema } from './ul-training.schema.js';

export interface TrainingSchedulerConfigSchema {
  enabled: boolean;
  type: 'constant' | 'step_decay' | 'cosine' | string;
  warmupSteps: number;
  stepSize: number;
  gamma: number;
  totalSteps: number;
  minLr: number;
}

export interface TrainingOptimizerConfigSchema {
  type: 'adam';
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  scheduler: TrainingSchedulerConfigSchema;
}

export interface TrainingGradientConfigSchema {
  maxNorm: number;
  accumSteps: number;
}

export interface TrainingLossScalingConfigSchema {
  enabled: boolean;
  initialScale: number;
  minScale: number;
  maxScale: number;
  scaleFactor: number;
  backoffFactor: number;
  growthInterval: number;
  overflowCheck: boolean;
}

export interface TrainingPrecisionConfigSchema {
  activations: 'f16' | 'f32';
  gradients: 'f16' | 'f32';
  loraParams: 'f16' | 'f32';
}

export interface TrainingAttentionConfigSchema {
  recomputeForward: boolean;
}

export interface TrainingTelemetryConfigSchema {
  mode: 'step' | 'window' | 'epoch';
  windowSize: number;
  emitNaNInfCounters: boolean;
  alerts: {
    enabled: boolean;
    failOnAlert: boolean;
    thresholds: {
      maxStepTimeMs: number | null;
      maxGradientNorm: number | null;
      maxNaNCount: number | null;
      maxInfCount: number | null;
      maxSaturationCount: number | null;
      minEffectiveLr: number | null;
    };
  };
}

export interface TrainingSettingsSchema {
  enabled: boolean;
  lora: LoraConfigSchema;
  optimizer: TrainingOptimizerConfigSchema;
  gradient: TrainingGradientConfigSchema;
  precision: TrainingPrecisionConfigSchema;
  attention: TrainingAttentionConfigSchema;
  telemetry: TrainingTelemetryConfigSchema;
  lossScaling: TrainingLossScalingConfigSchema;
  ul: UlTrainingConfigSchema;
}

export declare const DEFAULT_TRAINING_OPTIMIZER_CONFIG: TrainingOptimizerConfigSchema;
export declare const DEFAULT_TRAINING_GRADIENT_CONFIG: TrainingGradientConfigSchema;
export declare const DEFAULT_TRAINING_LOSS_SCALING_CONFIG: TrainingLossScalingConfigSchema;
export declare const DEFAULT_TRAINING_PRECISION_CONFIG: TrainingPrecisionConfigSchema;
export declare const DEFAULT_TRAINING_ATTENTION_CONFIG: TrainingAttentionConfigSchema;
export declare const DEFAULT_TRAINING_TELEMETRY_CONFIG: TrainingTelemetryConfigSchema;
export declare const DEFAULT_TRAINING_SETTINGS: TrainingSettingsSchema;
