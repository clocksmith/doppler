export type UlTrainingStage = 'stage1_joint' | 'stage2_base';

export interface UlNoiseScheduleSchema {
  type: string;
  minLogSNR: number;
  maxLogSNR: number;
  steps: number;
}

export interface UlPriorAlignmentSchema {
  enabled: boolean;
  weight: number;
}

export interface UlDecoderSigmoidWeightSchema {
  enabled: boolean;
  slope: number;
  midpoint: number;
}

export interface UlFreezeGroupsSchema {
  encoder: boolean;
  prior: boolean;
  decoder: boolean;
  base: boolean;
  lora: boolean;
}

export interface UlLossWeightsSchema {
  ce: number;
  prior: number;
  decoder: number;
  recon: number;
}

export interface UlTrainingConfigSchema {
  schemaVersion: number;
  enabled: boolean;
  stage: UlTrainingStage;
  lambda0: number;
  seed: number;
  artifactDir: string | null;
  stage1Artifact: string | null;
  stage1ArtifactHash: string | null;
  noiseSchedule: UlNoiseScheduleSchema;
  priorAlignment: UlPriorAlignmentSchema;
  decoderSigmoidWeight: UlDecoderSigmoidWeightSchema;
  lossWeights: UlLossWeightsSchema;
  freeze: UlFreezeGroupsSchema;
}

export declare const UL_STAGE_VALUES: readonly UlTrainingStage[];
export declare const UL_TRAINING_SCHEMA_VERSION: number;
export declare const DEFAULT_UL_NOISE_SCHEDULE: UlNoiseScheduleSchema;
export declare const DEFAULT_UL_PRIOR_ALIGNMENT: UlPriorAlignmentSchema;
export declare const DEFAULT_UL_DECODER_SIGMOID_WEIGHT: UlDecoderSigmoidWeightSchema;
export declare const DEFAULT_UL_FREEZE_GROUPS: UlFreezeGroupsSchema;
export declare const DEFAULT_UL_TRAINING_CONFIG: UlTrainingConfigSchema;

export declare function validateUlTrainingConfig(config: UlTrainingConfigSchema): UlTrainingConfigSchema;
