export type DistillTrainingStage = 'stage_a' | 'stage_b';
export type DistillStudentGraphMode = 'projection_head' | 'transformer_full';

export interface DistillFreezeConfigSchema {
  encoder: boolean;
  prior: boolean;
  decoder: boolean;
  base: boolean;
  lora: boolean;
}

export interface DistillTrainingConfigSchema {
  schemaVersion: number;
  enabled: boolean;
  stage: DistillTrainingStage;
  teacherModelId: string | null;
  studentModelId: string | null;
  datasetId: string | null;
  datasetPath: string | null;
  languagePair: string | null;
  sourceLangs: string[] | null;
  targetLangs: string[] | null;
  pairAllowlist: string[] | null;
  strictPairContract: boolean;
  shardIndex: number | null;
  shardCount: number | null;
  resumeFrom: string | null;
  artifactDir: string | null;
  stageAArtifact: string | null;
  stageAArtifactHash: string | null;
  temperature: number;
  alphaKd: number;
  alphaCe: number;
  allowHintFallback: boolean;
  tripletMargin: number;
  studentGraphMode: DistillStudentGraphMode | null;
  freeze: DistillFreezeConfigSchema;
}

export declare const DISTILL_STAGE_VALUES: readonly DistillTrainingStage[];
export declare const DISTILL_STUDENT_GRAPH_MODE_VALUES: readonly DistillStudentGraphMode[];
export declare const DISTILL_TRAINING_SCHEMA_VERSION: number;
export declare const DEFAULT_DISTILL_FREEZE_GROUPS: DistillFreezeConfigSchema;
export declare const DEFAULT_DISTILL_TRAINING_CONFIG: DistillTrainingConfigSchema;

export declare function validateDistillTrainingConfig(
  config: DistillTrainingConfigSchema
): DistillTrainingConfigSchema;
