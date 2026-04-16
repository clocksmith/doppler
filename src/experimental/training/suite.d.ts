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
      dataScope?: {
        sourceLangs: string[] | null;
        targetLangs: string[] | null;
        pairAllowlist: string[] | null;
        strictPairContract: boolean;
      } | null;
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
  runtimeProfile?: string | null;
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

export interface DistillDataScope {
  sourceLangs: string[] | null;
  targetLangs: string[] | null;
  pairAllowlist: string[] | null;
  sourceLangSet: Set<string> | null;
  targetLangSet: Set<string> | null;
  pairAllowlistSet: Set<string> | null;
  strictPairContract: boolean;
}

export interface DistillDatasetReport {
  absolutePath: string;
  rowCount: number;
  sampleCount: number;
  directionCounts: Record<string, number>;
  dataScope: {
    sourceLangs: string[] | null;
    targetLangs: string[] | null;
    pairAllowlist: string[] | null;
    strictPairContract: boolean;
  } | null;
  shardCount?: number;
  shardPaths?: string[];
  createDataset(options?: Record<string, unknown>): {
    batches(): AsyncGenerator<Record<string, unknown>, void, unknown>;
  };
}

export interface DistillRuntimeContext {
  stage: 'stage_a' | 'stage_b';
  teacherPipeline: Record<string, unknown>;
  studentPipeline: Record<string, unknown>;
  teacherModelId: string;
  studentModelId: string;
  teacherModelUrl: string | null;
  studentModelUrl: string | null;
  topK: number;
  temperature: number;
  alphaKd: number;
  alphaCe: number;
  tripletMargin: number;
  studentGraphMode: string;
  targetTokenMode: string;
  cleanup(): Promise<void>;
}

export interface DistillStudentFixture {
  config: Record<string, unknown>;
  model: {
    forward: (input: unknown, tape: unknown) => Promise<unknown>;
    forwardDistill?: (batch: unknown, tape: unknown, options?: Record<string, unknown>) => Promise<{ logits: unknown }>;
    cleanupDistillStep?: () => void;
    loraParams?: () => unknown[];
    paramGroups?: () => Record<string, unknown[]>;
  };
  outputDim?: number;
  embeddingDim?: number;
  cleanup(): void;
}

export declare const trainingHarness: TrainingHarness;

export declare function runTrainingSuite(
  options?: RunTrainingSuiteOptions
): Promise<TrainingSuiteSummary>;

export declare function runTrainingBenchSuite(
  options?: RunTrainingSuiteOptions
): Promise<TrainingBenchSuiteResult>;

export declare function resolveDistillDataScope(
  options?: RunTrainingSuiteOptions,
  trainingConfig?: Record<string, unknown> | null
): DistillDataScope;

export declare function buildDistillPrompt(sample: Record<string, unknown>): string;

export declare function normalizeDistillStudentGraphMode(value: unknown): string;

export declare function loadDistillDatasetFromJsonl(
  datasetPath: string,
  scopeOptions?: DistillDataScope | null
): Promise<DistillDatasetReport | null>;

export declare function loadDistillModelHandle(
  modelRef: string,
  role: string,
  loadOptions?: Record<string, unknown>
): Promise<{
  modelRef: string;
  modelUrl: string | null;
  manifest: Record<string, unknown>;
  pipeline: Record<string, unknown>;
}>;

export declare function createDistillRuntimeContext(
  options?: RunTrainingSuiteOptions,
  trainingConfig?: Record<string, unknown> | null
): Promise<DistillRuntimeContext>;

export declare function createToyModelFixture(
  overrides?: Record<string, unknown>
): {
  config: Record<string, unknown>;
  model: {
    forward: (input: unknown, tape: unknown) => Promise<unknown>;
    loraParams(): unknown[];
    paramGroups(): Record<string, unknown[]>;
  };
  batch: Record<string, unknown>;
  cleanup(): void;
};

export declare function createDistillStudentRuntimeModelFixture(
  overrides?: Record<string, unknown>,
  options?: Record<string, unknown>
): Promise<DistillStudentFixture>;

export declare function buildDistillTrainingOverrides(
  options?: RunTrainingSuiteOptions
): Record<string, unknown> | null;
