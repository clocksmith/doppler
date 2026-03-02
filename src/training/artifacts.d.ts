import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { DistillTrainingStage } from '../config/schema/distill-training.schema.js';
import type { UlTrainingStage } from '../config/schema/ul-training.schema.js';

export interface UlTrainingContract {
  enabled: boolean;
  stage: UlTrainingStage | null;
  artifactDir: string | null;
  stage1Artifact: string | null;
  stage1ArtifactHash: string | null;
}

export interface DistillTrainingContract {
  enabled: boolean;
  stage: DistillTrainingStage | null;
  artifactDir: string | null;
  stageAArtifact: string | null;
  stageAArtifactHash: string | null;
  teacherModelId: string | null;
  studentModelId: string | null;
  datasetId: string | null;
  datasetPath: string | null;
  languagePair: string | null;
}

export interface UlArtifactFinalizeResult {
  kind: 'ul';
  stage: UlTrainingStage;
  runDir: string;
  metricsPath: string;
  manifestPath: string;
  manifestHash: string;
  manifestContentHash: string;
  manifestFileHash: string;
  stage1Dependency: {
    path: string;
    hash: string;
    manifestHash: string | null;
  } | null;
}

export interface DistillArtifactFinalizeResult {
  kind: 'distill';
  stage: DistillTrainingStage;
  runDir: string;
  metricsPath: string;
  manifestPath: string;
  manifestHash: string;
  manifestContentHash: string;
  manifestFileHash: string;
  stageADependency: {
    path: string;
    hash: string;
    manifestHash: string | null;
    metricsHash: string;
    metricsSummary: {
      stepCount: number;
      kdCount: number;
      tripletCount: number;
      kdMean: number | null;
      tripletMean: number | null;
      totalLossMean: number | null;
    };
  } | null;
}

export interface UlArtifactSession {
  appendStep(entry: Record<string, unknown>): Promise<void>;
  finalize(stepMetrics: Record<string, unknown>[]): Promise<UlArtifactFinalizeResult>;
}

export interface DistillArtifactSession {
  appendStep(entry: Record<string, unknown>): Promise<void>;
  finalize(stepMetrics: Record<string, unknown>[]): Promise<DistillArtifactFinalizeResult>;
}

export interface CreateUlArtifactSessionOptions {
  config: TrainingConfigSchema;
  stage: UlTrainingStage;
  runOptions?: Record<string, unknown>;
}

export interface CreateDistillArtifactSessionOptions {
  config: TrainingConfigSchema;
  stage: DistillTrainingStage;
  runOptions?: Record<string, unknown>;
}

export interface Stage1ArtifactContext {
  manifestPath: string;
  manifestHash: string;
  ulContractHash: string | null;
  latentDataset: {
    path: string;
    hash: string;
    count: number;
    summary: {
      lambdaMean: number;
      noisyStdMean: number;
      cleanStdMean: number;
      noiseStdMean: number;
      scheduleMaxStep: number;
      vectorCount: number;
    };
    entries: Record<string, unknown>[];
  };
}

export interface StageAArtifactContext {
  manifestPath: string;
  manifestHash: string;
  distillContractHash: string | null;
  metrics: {
    path: string;
    hash: string;
    count: number;
    summary: {
      stepCount: number;
      kdCount: number;
      tripletCount: number;
      kdMean: number | null;
      tripletMean: number | null;
      totalLossMean: number | null;
    };
    entries: Record<string, unknown>[];
  };
  metricsSummary: {
    stepCount: number;
    kdCount: number;
    tripletCount: number;
    kdMean: number | null;
    tripletMean: number | null;
    totalLossMean: number | null;
  };
}

export declare function resolveUlTrainingContract(
  ulConfig: TrainingConfigSchema['training']['ul'] | null | undefined
): UlTrainingContract;

export declare function resolveDistillTrainingContract(
  distillConfig: TrainingConfigSchema['training']['distill'] | null | undefined
): DistillTrainingContract;

export declare function createUlArtifactSession(
  options: CreateUlArtifactSessionOptions
): Promise<UlArtifactSession | null>;

export declare function createDistillArtifactSession(
  options: CreateDistillArtifactSessionOptions
): Promise<DistillArtifactSession | null>;

export declare function resolveStage1ArtifactContext(
  config: TrainingConfigSchema
): Promise<Stage1ArtifactContext | null>;

export declare function resolveStageAArtifactContext(
  config: TrainingConfigSchema
): Promise<StageAArtifactContext | null>;
