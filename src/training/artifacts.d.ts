import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { UlTrainingStage } from '../config/schema/ul-training.schema.js';

export interface UlTrainingContract {
  enabled: boolean;
  stage: UlTrainingStage | null;
  artifactDir: string | null;
  stage1Artifact: string | null;
  stage1ArtifactHash: string | null;
}

export interface UlArtifactFinalizeResult {
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

export interface UlArtifactSession {
  appendStep(entry: Record<string, unknown>): Promise<void>;
  finalize(stepMetrics: Record<string, unknown>[]): Promise<UlArtifactFinalizeResult>;
}

export interface CreateUlArtifactSessionOptions {
  config: TrainingConfigSchema;
  stage: UlTrainingStage;
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

export declare function resolveUlTrainingContract(
  ulConfig: TrainingConfigSchema['training']['ul'] | null | undefined
): UlTrainingContract;

export declare function createUlArtifactSession(
  options: CreateUlArtifactSessionOptions
): Promise<UlArtifactSession | null>;

export declare function resolveStage1ArtifactContext(
  config: TrainingConfigSchema
): Promise<Stage1ArtifactContext | null>;
