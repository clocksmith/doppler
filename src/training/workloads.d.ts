export type TrainingWorkloadKind = 'lora' | 'distill' | 'ul';
export type TrainingSurfaceSupport = 'node' | 'browser' | 'both';
export type TrainingSelectionGoal = 'max' | 'min';
export type TrainingEvalKind =
  | 'translation'
  | 'text_generation'
  | 'classification'
  | 'retrieval'
  | 'custom';

export interface TrainingEvalDataset {
  id: string;
  datasetPath: string;
  evalKind: TrainingEvalKind;
  metrics: string[];
  sourceLangs: string[] | null;
  targetLangs: string[] | null;
  pairAllowlist: string[] | null;
}

export interface TrainingOptimizerConfig {
  type: string;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  scheduler: {
    enabled: boolean;
    type: string;
    warmupSteps: number;
    stepSize: number;
    gamma: number;
    totalSteps: number;
    minLr: number;
  };
}

export interface TrainingWorkloadTrainingConfig {
  optimizer: TrainingOptimizerConfig;
  batchSize: number;
  accumSteps: number;
  steps: number;
  precision: {
    activations: string;
    gradients: string;
    loraParams: string;
  };
  gradientClipping: {
    maxNorm: number;
  };
}

export interface LoRAWorkloadPipelineConfig {
  datasetFormat: string;
  taskType: string;
  adapter: {
    rank: number;
    alpha: number;
    dropout: number;
    targetModules: string[];
  };
  freeze: {
    encoder: boolean;
    prior: boolean;
    decoder: boolean;
    base: boolean;
    lora: boolean;
  };
  export: {
    enabled: boolean;
    atCheckpoints: boolean;
    select: string;
    id: string | null;
    name: string | null;
    format: string;
  } | null;
  activation: {
    enabled: boolean;
    autoActivate: boolean;
    smokePrompt: string | null;
  } | null;
}

export interface DistillStagePlanEntry {
  id: string;
  trainingStage: string;
  objective: string;
  steps: number;
  checkpointEvery: number;
  selectionMetric: string;
  selectionGoal: TrainingSelectionGoal;
  evalSchedule: string;
}

export interface DistillWorkloadPipelineConfig {
  stagePlan: DistillStagePlanEntry[];
  studentGraphMode: string;
  temperature: number;
  alphaKd: number;
  alphaCe: number;
  tripletMargin: number;
  sourceLangs: string[] | null;
  targetLangs: string[] | null;
  pairAllowlist: string[] | null;
  strictPairContract: boolean;
  subsetSpec: Record<string, unknown> | null;
}

export interface TrainingWorkloadPack {
  schemaVersion: number;
  kind: TrainingWorkloadKind;
  id: string;
  description: string;
  claimBoundary: string;
  seed: number;
  baseModelId: string;
  studentModelId: string | null;
  teacherModelId: string | null;
  datasetId: string;
  datasetPath: string | null;
  evalDatasets: TrainingEvalDataset[];
  trainingSchemaVersion: number;
  checkpointEvery: number;
  selectionMetric: string;
  selectionGoal: TrainingSelectionGoal;
  surfaceSupport: TrainingSurfaceSupport;
  training: TrainingWorkloadTrainingConfig;
  pipeline: LoRAWorkloadPipelineConfig | DistillWorkloadPipelineConfig | Record<string, unknown>;
  configHash: string;
}

export interface LoadedTrainingWorkload {
  absolutePath: string;
  path: string;
  raw: string;
  workloadSha256: string;
  workload: TrainingWorkloadPack;
}

export declare const TRAINING_WORKLOAD_SCHEMA_VERSION: number;
export declare const TRAINING_WORKLOAD_KINDS: readonly TrainingWorkloadKind[];
export declare const TRAINING_WORKLOAD_SURFACE_SUPPORT: readonly TrainingSurfaceSupport[];
export declare const TRAINING_SELECTION_GOALS: readonly TrainingSelectionGoal[];
export declare const TRAINING_EVAL_KINDS: readonly TrainingEvalKind[];

export declare function normalizeTrainingWorkloadPack(
  payload: Record<string, unknown>,
  context?: { label?: string }
): TrainingWorkloadPack;

export declare function loadTrainingWorkloadPack(
  input: string,
  options?: { registryPath?: string | null }
): Promise<LoadedTrainingWorkload>;

export declare function serializeTrainingWorkloadLock(
  loadedWorkload: LoadedTrainingWorkload
): string;
