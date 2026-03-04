import type { ConverterConfigSchema } from '../config/schema/converter.schema.js';

export type ToolingCommand = 'convert' | 'debug' | 'bench' | 'verify';
export type ToolingSurface = 'browser' | 'node';
export type ToolingSuite = 'kernels' | 'inference' | 'training' | 'bench' | 'debug' | 'diffusion' | 'energy';
export type ToolingIntent = 'verify' | 'investigate' | 'calibrate' | null;
export type ToolingTrainingStage = 'stage1_joint' | 'stage2_base' | 'stage_a' | 'stage_b';

export interface ToolingConvertExecutionPayload {
  workers?: number | null;
  workerCountPolicy?: 'cap' | 'error' | null;
  maxInFlightJobs?: number | null;
  rowChunkRows?: number | null;
  rowChunkMinTensorBytes?: number | null;
  useGpuCast?: boolean | null;
  gpuCastMinTensorBytes?: number | null;
  [key: string]: unknown;
}

export interface ToolingConvertPayload {
  converterConfig: Partial<ConverterConfigSchema>;
  execution?: ToolingConvertExecutionPayload | null;
  [key: string]: unknown;
}

export interface ToolingCommandRequestInput {
  command: ToolingCommand;
  suite?: ToolingSuite;
  modelId?: string;
  trainingTests?: string[];
  trainingStage?: ToolingTrainingStage;
  trainingConfig?: Record<string, unknown>;
  stage1Artifact?: string;
  stage1ArtifactHash?: string;
  ulArtifactDir?: string;
  stageAArtifact?: string;
  stageAArtifactHash?: string;
  distillArtifactDir?: string;
  teacherModelId?: string;
  studentModelId?: string;
  distillDatasetId?: string;
  distillDatasetPath?: string;
  distillLanguagePair?: string;
  distillShardIndex?: number;
  distillShardCount?: number;
  resumeFrom?: string;
  forceResume?: boolean;
  forceResumeReason?: string;
  forceResumeSource?: string;
  checkpointOperator?: string;
  trainingSchemaVersion?: number;
  trainingBenchSteps?: number;
  workloadType?: string;
  modelUrl?: string;
  cacheMode?: 'cold' | 'warm';
  loadMode?: 'opfs' | 'http' | 'memory';
  runtimePreset?: string;
  runtimeConfigUrl?: string;
  runtimeConfig?: Record<string, unknown>;
  inputDir?: string;
  outputDir?: string;
  convertPayload?: ToolingConvertPayload;
  captureOutput?: boolean;
  keepPipeline?: boolean;
  report?: Record<string, unknown> | null;
  timestamp?: string | Date | null;
  searchParams?: URLSearchParams | null;
}

export interface ToolingCommandRequest {
  command: ToolingCommand;
  suite: ToolingSuite | null;
  intent: ToolingIntent;
  modelId: string | null;
  trainingTests: string[] | null;
  trainingStage: ToolingTrainingStage | null;
  trainingConfig: Record<string, unknown> | null;
  stage1Artifact: string | null;
  stage1ArtifactHash: string | null;
  ulArtifactDir: string | null;
  stageAArtifact: string | null;
  stageAArtifactHash: string | null;
  distillArtifactDir: string | null;
  teacherModelId: string | null;
  studentModelId: string | null;
  distillDatasetId: string | null;
  distillDatasetPath: string | null;
  distillLanguagePair: string | null;
  distillShardIndex: number | null;
  distillShardCount: number | null;
  resumeFrom: string | null;
  forceResume: boolean | null;
  forceResumeReason: string | null;
  forceResumeSource: string | null;
  checkpointOperator: string | null;
  trainingSchemaVersion: number | null;
  trainingBenchSteps: number | null;
  workloadType: string | null;
  modelUrl: string | null;
  cacheMode: 'cold' | 'warm' | null;
  loadMode: 'opfs' | 'http' | 'memory' | null;
  runtimePreset: string | null;
  runtimeConfigUrl: string | null;
  runtimeConfig: Record<string, unknown> | null;
  inputDir: string | null;
  outputDir: string | null;
  convertPayload: ToolingConvertPayload | null;
  captureOutput: boolean;
  keepPipeline: boolean;
  report: Record<string, unknown> | null;
  timestamp: string | Date | null;
  searchParams: URLSearchParams | null;
}

export declare const TOOLING_COMMANDS: readonly ToolingCommand[];
export declare const TOOLING_SURFACES: readonly ToolingSurface[];
export declare const TOOLING_SUITES: readonly ToolingSuite[];
export declare const TOOLING_VERIFY_SUITES: readonly Exclude<ToolingSuite, 'bench' | 'debug'>[];
export declare const TOOLING_TRAINING_COMMAND_SCHEMA_VERSION: number;

export declare function normalizeToolingCommandRequest(
  input: ToolingCommandRequestInput
): ToolingCommandRequest;

export declare function buildRuntimeContractPatch(
  commandRequest: ToolingCommandRequestInput
): {
  shared: {
    harness: { mode: ToolingSuite; modelId: string | null };
    tooling: { intent: Exclude<ToolingIntent, null> };
  };
} | null;

export declare function ensureCommandSupportedOnSurface(
  commandRequest: ToolingCommandRequestInput,
  surface: ToolingSurface
): { request: ToolingCommandRequest; surface: ToolingSurface };
