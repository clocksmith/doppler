export interface TrainingSuiteTestResult {
  name: string;
  passed: boolean;
  skipped?: boolean;
  duration: number;
  error?: string;
  metrics?: Record<string, unknown>;
  artifact?: Record<string, unknown>;
}

export interface TrainingSuiteSummary {
  suite: 'training';
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  results: TrainingSuiteTestResult[];
  modelId: string;
  metrics: {
    testsRun: number;
    selectedTests: string[];
    availableTests: string[];
    trainingStage: 'stage1_joint' | 'stage2_base' | 'stage_a' | 'stage_b' | null;
    trainingSchemaVersion: number;
    adapterActivation?: {
      activated: boolean;
      adapterName: string | null;
      source: string | null;
      reason: string | null;
    } | null;
  };
  deviceInfo: Record<string, unknown> | null;
}

export interface TrainingBenchSuiteResult {
  suite: 'bench';
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  results: TrainingSuiteTestResult[];
  modelId: string;
  metrics: {
    workloadType: 'training';
    warmupRuns: number;
    timedRuns: number;
    completedTimedRuns: number;
    stepsPerRun: number;
    trainingSchemaVersion: number;
    trainingMetricsReport: Record<string, unknown>[];
    progress: {
      shardIndex: number;
      shardCount: number;
      stepsPerShard: number | null;
      completedGlobalSteps: number | null;
      totalGlobalSteps: number | null;
      percentComplete: number | null;
      etaMs: number | null;
      etaIso: string | null;
      elapsedMs: number;
      updatedAt: string;
    };
    ulArtifacts: Record<string, unknown>[];
    distillArtifacts: Record<string, unknown>[];
    adapterExports?: Array<{
      runIndex: number;
      id: string | null;
      name: string | null;
      hash: string;
    }>;
    adapterActivation?: {
      activated: boolean;
      adapterName: string | null;
      source: string | null;
      reason: string | null;
    } | null;
    checkpointResumeTimeline: Array<Record<string, unknown>>;
    distillDataset?: {
      path: string;
      rowCount: number;
      sampleCount: number;
      shardCount?: number;
      directionCounts: Record<string, number>;
    } | null;
    latency: {
      runMs: Record<string, unknown>;
      stepMs: Record<string, unknown>;
    };
    throughput: {
      stepsPerSec: Record<string, unknown>;
    };
  };
  deviceInfo: Record<string, unknown> | null;
}

export interface TrainingHarness {
  getGPU(): Promise<boolean>;
  runTest(
    name: string,
    options?: RunTrainingSuiteOptions
  ): Promise<{
    passed: boolean;
    skipped?: boolean;
    error?: string;
    metrics?: Record<string, unknown>;
    artifact?: Record<string, unknown>;
  }>;
  listTests(): string[];
}

export interface RunTrainingSuiteOptions {
  command?: string;
  surface?: string;
  modelId?: string;
  modelUrl?: string;
  runtimePreset?: string | null;
  workloadType?: string;
  trainingTests?: string[];
  trainingStage?: 'stage1_joint' | 'stage2_base' | 'stage_a' | 'stage_b';
  trainingConfig?: Record<string, unknown>;
  trainingSchemaVersion?: number;
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
  checkpointOperator?: string | null;
  trainingBenchSteps?: number;
  benchRun?: Record<string, unknown> | null;
  adapterActivation?: {
    enabled?: boolean;
    autoActivate?: boolean;
    adapter?: unknown;
    adapterManifest?: Record<string, unknown>;
    adapterManifestJson?: string;
    adapterManifestUrl?: string;
    adapterManifestPath?: string;
    export?: {
      id: string;
      name: string;
      baseModel: string;
      rank: number;
      alpha: number;
      targetModules: string[];
      tensors: Array<{
        name: string;
        paramIndex: number;
      }>;
      format?: 'base64' | 'array';
      pretty?: boolean;
    };
  };
  checkpointEvery?: number;
  timestamp?: string | Date;
}

export declare const trainingHarness: TrainingHarness;

export declare function runTrainingSuite(
  options?: RunTrainingSuiteOptions
): Promise<TrainingSuiteSummary>;

export declare function runTrainingBenchSuite(
  options?: RunTrainingSuiteOptions
): Promise<TrainingBenchSuiteResult>;
