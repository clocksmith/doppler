import type { LoraConfigSchema } from './lora.schema.js';

export interface TrainingOptimizerConfigSchema {
  type: 'adam';
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
}

export interface TrainingGradientConfigSchema {
  maxNorm: number;
  accumSteps: number;
}

export interface TrainingPrecisionConfigSchema {
  activations: 'f16' | 'f32';
  gradients: 'f16' | 'f32';
  loraParams: 'f16' | 'f32';
}

export interface TrainingAttentionConfigSchema {
  recomputeForward: boolean;
}

export interface TrainingSettingsSchema {
  enabled: boolean;
  lora: LoraConfigSchema;
  optimizer: TrainingOptimizerConfigSchema;
  gradient: TrainingGradientConfigSchema;
  precision: TrainingPrecisionConfigSchema;
  attention: TrainingAttentionConfigSchema;
}

export declare const DEFAULT_TRAINING_OPTIMIZER_CONFIG: TrainingOptimizerConfigSchema;
export declare const DEFAULT_TRAINING_GRADIENT_CONFIG: TrainingGradientConfigSchema;
export declare const DEFAULT_TRAINING_PRECISION_CONFIG: TrainingPrecisionConfigSchema;
export declare const DEFAULT_TRAINING_ATTENTION_CONFIG: TrainingAttentionConfigSchema;
export declare const DEFAULT_TRAINING_SETTINGS: TrainingSettingsSchema;
