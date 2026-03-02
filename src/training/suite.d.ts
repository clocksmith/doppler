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
    ulArtifacts: Record<string, unknown>[];
    distillArtifacts: Record<string, unknown>[];
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
  modelId?: string;
  modelUrl?: string;
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
  distillLanguagePair?: string;
  trainingBenchSteps?: number;
  benchRun?: Record<string, unknown> | null;
  timestamp?: string | Date;
}

export declare const trainingHarness: TrainingHarness;

export declare function runTrainingSuite(
  options?: RunTrainingSuiteOptions
): Promise<TrainingSuiteSummary>;

export declare function runTrainingBenchSuite(
  options?: RunTrainingSuiteOptions
): Promise<TrainingBenchSuiteResult>;
