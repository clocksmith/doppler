import type { LoadedTrainingWorkload } from './workloads.js';

export declare const LORA_RUNNER_SUPPORT_CONTRACT: Readonly<{
  supportedBaseModelId: string;
  supportedDatasetFormat: string;
  registeredBaseModelIds: readonly string[];
  registeredDatasetFormats: readonly string[];
  implementedRunnerKeys: readonly string[];
}>;

export declare const LORA_RUNNER_BASE_MODEL_REGISTRY: Readonly<Record<string, Readonly<{
  baseModelId: string;
  modelRef?: string;
  family: string;
  runnerKind: string;
  requiresExternalTrainer?: boolean;
}>>>;

export declare const LORA_RUNNER_DATASET_FORMAT_REGISTRY: Readonly<Record<string, Readonly<{
  datasetFormat: string;
  datasetKind: string;
}>>>;

export declare function getLoraRunnerCompatibility(workload: unknown): {
  schemaVersion: 1;
  supported: boolean;
  runnerContract: typeof LORA_RUNNER_SUPPORT_CONTRACT;
  observed: {
    baseModelId: string;
    datasetFormat: string;
    taskType: string;
    runnerKey: string;
    baseModelFamily: string | null;
    baseModelRunnerKind: string | null;
    requiresExternalTrainer: boolean;
    datasetKind: string | null;
    registeredBaseModel: boolean;
    registeredDatasetFormat: boolean;
  };
  blockedReasons: string[];
};

export declare function assertLoraRunnerCompatibility(workload: unknown): {
  schemaVersion: 1;
  supported: true;
  runnerContract: typeof LORA_RUNNER_SUPPORT_CONTRACT;
  observed: {
    baseModelId: string;
    datasetFormat: string;
    taskType: string;
    runnerKey: string;
    baseModelFamily: string | null;
    baseModelRunnerKind: string | null;
    requiresExternalTrainer: boolean;
    datasetKind: string | null;
    registeredBaseModel: boolean;
    registeredDatasetFormat: boolean;
  };
  blockedReasons: [];
};

export declare function preflightCausalLmLoraWorkload(
  workload: unknown,
  options?: {
    datasetPath?: string;
    fetch?: (url: string) => Promise<string>;
    readFile?: (path: string) => Promise<string>;
  }
): Promise<{
  schemaVersion: 1;
  supported: boolean;
  runnerKey: string;
  baseModelId: string;
  baseModelFamily: string | null;
  datasetPath: string;
  datasetFormat: string;
  taskType: string;
  rowCount: number;
  firstRowId: string | null;
  lastRowId: string | null;
  textPairFields: {
    prompt: string | null;
    completion: string | null;
  };
  textPairLengths: {
    minPromptChars: number;
    maxPromptChars: number;
    minCompletionChars: number;
    maxCompletionChars: number;
  };
  adapter: {
    rank: number;
    alpha: number;
    targetModules: string[];
  };
  blockedReasons: string[];
}>;

export interface CausalLmLoraTrainerTensor {
  name: string;
  shape: [number, number];
  tensor?: Float32Array | GPUBuffer | { buffer?: GPUBuffer; shape?: [number, number]; dtype?: string };
  data?: Float32Array | number[];
  values?: Float32Array | number[];
  dtype?: 'f16' | 'f32';
}

export interface CausalLmLoraTrainerResult {
  checkpointId?: string | null;
  checkpointStep?: number | null;
  adapterId?: string | null;
  adapterName?: string | null;
  trainerId?: string | null;
  runnerId?: string | null;
  metrics?: Record<string, unknown>;
  receipts?: unknown[];
  evalReports?: Array<Record<string, unknown>>;
  tensors?: CausalLmLoraTrainerTensor[];
  weights?: CausalLmLoraTrainerTensor[];
}

export interface CausalLmLoraTrainerInput {
  schemaVersion: 1;
  runnerKind: 'causal_lm_lora';
  workload: LoadedTrainingWorkload['workload'];
  loadedWorkload: LoadedTrainingWorkload;
  compatibility: ReturnType<typeof getLoraRunnerCompatibility>;
  preflight: Awaited<ReturnType<typeof preflightCausalLmLoraWorkload>>;
  dataset: {
    absolutePath: string;
    rowCount: number;
    rows: Array<Record<string, unknown>>;
    datasetHash: string;
  };
  adapter: Record<string, unknown>;
  training: Record<string, unknown>;
  export: Record<string, unknown> | null;
  layout: Record<string, string>;
}

export type CausalLmLoraTrainer = (input: CausalLmLoraTrainerInput) => Promise<CausalLmLoraTrainerResult> | CausalLmLoraTrainerResult;

export declare function runLoraPipeline(options: {
  loadedWorkload: LoadedTrainingWorkload;
  runRoot?: string | null;
  timestamp?: string | Date | null;
  causalLmTrainer?: CausalLmLoraTrainer;
  fetch?: (url: string) => Promise<string>;
  readFile?: (path: string) => Promise<string>;
}): Promise<Record<string, unknown>>;

export declare function evaluateLoraCheckpoint(options: {
  loadedWorkload: LoadedTrainingWorkload;
  checkpointPath: string;
  checkpointId?: string | null;
  checkpointStep?: number | null;
  layout?: Record<string, string> | null;
}): Promise<Record<string, unknown>[]>;

export declare function exportLoraCheckpoint(options: {
  loadedWorkload: LoadedTrainingWorkload;
  checkpointPath: string;
  checkpointId?: string | null;
  checkpointStep?: number | null;
  layout?: Record<string, string> | null;
  exportsDir?: string | null;
  datasetHash?: string | null;
}): Promise<Record<string, unknown>>;

export declare function watchLoraCheckpoints(options: {
  loadedWorkload: LoadedTrainingWorkload;
  runRoot: string;
  pollIntervalMs?: number | null;
  stopWhenIdle?: boolean;
}): Promise<{ ok: true; processedCount: number; manifestPath: string }>;

export declare function compareLoraRun(options: {
  runRoot: string;
}): Promise<Record<string, unknown>>;

export declare function qualityGateLoraRun(options: {
  runRoot: string;
}): Promise<Record<string, unknown>>;
