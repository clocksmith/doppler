import type { ConverterConfigSchema } from '../config/schema/converter.schema.js';

export type ToolingCommand = 'convert' | 'debug' | 'bench' | 'verify' | 'lora' | 'distill' | 'diagnose';
export type ToolingSurface = 'browser' | 'node';
export type ToolingWorkload = 'kernels' | 'inference' | 'embedding' | 'training' | 'diffusion' | 'energy';
export type ToolingIntent = 'verify' | 'investigate' | 'calibrate' | null;
export type ToolingTrainingStage = 'stage1_joint' | 'stage2_base' | 'stage_a' | 'stage_b';
export type ToolingDistillAction = 'run' | 'stage-a' | 'stage-b' | 'eval' | 'watch' | 'compare' | 'quality-gate' | 'subsets';
export type ToolingLoraAction = 'run' | 'eval' | 'watch' | 'export' | 'compare' | 'quality-gate' | 'activate';

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
  action?: ToolingDistillAction | ToolingLoraAction;
  workload?: ToolingWorkload;
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
  distillSourceLangs?: string[];
  distillTargetLangs?: string[];
  distillPairAllowlist?: string[];
  strictPairContract?: boolean;
  distillShardIndex?: number;
  distillShardCount?: number;
  resumeFrom?: string;
  forceResume?: boolean;
  forceResumeReason?: string;
  forceResumeSource?: string;
  checkpointOperator?: string;
  trainingSchemaVersion?: number;
  trainingBenchSteps?: number;
  checkpointEvery?: number;
  workloadType?: string;
  modelUrl?: string;
  cacheMode?: 'cold' | 'warm';
  loadMode?: 'opfs' | 'http' | 'memory';
  runtimeProfile?: string;
  runtimeConfigUrl?: string;
  runtimeConfig?: Record<string, unknown>;
  inputDir?: string;
  outputDir?: string;
  convertPayload?: ToolingConvertPayload;
  workloadPath?: string;
  runRoot?: string;
  checkpointPath?: string;
  checkpointId?: string;
  checkpointStep?: number;
  stageId?: string;
  stageArtifact?: string;
  subsetManifest?: string;
  evalDatasetId?: string;
  pollIntervalMs?: number;
  stopWhenIdle?: boolean;
  captureOutput?: boolean;
  keepPipeline?: boolean;
  report?: Record<string, unknown> | null;
  timestamp?: string | Date | null;
  searchParams?: URLSearchParams | null;
  baselineProvider?: string | null;
  observedProvider?: string | null;
}

export interface ToolingCommandRequest {
  command: ToolingCommand;
  workload: ToolingWorkload | null;
  intent: ToolingIntent;
  action: ToolingDistillAction | ToolingLoraAction | null;
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
  distillSourceLangs: string[] | null;
  distillTargetLangs: string[] | null;
  distillPairAllowlist: string[] | null;
  strictPairContract: boolean | null;
  distillShardIndex: number | null;
  distillShardCount: number | null;
  resumeFrom: string | null;
  forceResume: boolean | null;
  forceResumeReason: string | null;
  forceResumeSource: string | null;
  checkpointOperator: string | null;
  trainingSchemaVersion: number | null;
  trainingBenchSteps: number | null;
  checkpointEvery: number | null;
  workloadType: string | null;
  modelUrl: string | null;
  cacheMode: 'cold' | 'warm' | null;
  loadMode: 'opfs' | 'http' | 'memory' | null;
  runtimeProfile: string | null;
  runtimeConfigUrl: string | null;
  runtimeConfig: Record<string, unknown> | null;
  inputDir: string | null;
  outputDir: string | null;
  convertPayload: ToolingConvertPayload | null;
  workloadPath: string | null;
  runRoot: string | null;
  checkpointPath: string | null;
  checkpointId: string | null;
  checkpointStep: number | null;
  stageId: string | null;
  stageArtifact: string | null;
  subsetManifest: string | null;
  evalDatasetId: string | null;
  pollIntervalMs: number | null;
  stopWhenIdle: boolean | null;
  captureOutput: boolean;
  keepPipeline: boolean;
  report: Record<string, unknown> | null;
  timestamp: string | Date | null;
  searchParams: URLSearchParams | null;
  baselineProvider: string | null;
  observedProvider: string | null;
}

export declare const TOOLING_COMMANDS: readonly ToolingCommand[];
export declare const TOOLING_SURFACES: readonly ToolingSurface[];
export declare const TOOLING_WORKLOADS: readonly ToolingWorkload[];
export declare const TOOLING_VERIFY_WORKLOADS: readonly ToolingWorkload[];
export declare const TOOLING_TRAINING_COMMAND_SCHEMA_VERSION: number;

export declare function normalizeToolingCommandRequest(
  input: ToolingCommandRequestInput
): ToolingCommandRequest;

export declare function buildRuntimeContractPatch(
  commandRequest: ToolingCommandRequestInput
): {
  shared: {
    harness: { mode: ToolingCommand; workload: ToolingWorkload; modelId: string | null };
    tooling: { intent: Exclude<ToolingIntent, null> };
  };
} | null;

export declare function ensureCommandSupportedOnSurface(
  commandRequest: ToolingCommandRequestInput,
  surface: ToolingSurface
): { request: ToolingCommandRequest; surface: ToolingSurface };
