export type RuntimeOptimizationWorkload = 'inference' | 'embedding' | 'rerank';
export type RuntimeOptimizationDirection = 'maximize' | 'minimize';
export type RuntimeOptimizationCandidateKind =
  | 'runtime_profile'
  | 'runtime-profile'
  | 'registered-kernel-variant'
  | 'registered-execution-graph-patch';

export interface RuntimeOptimizationContract {
  schema: 'doppler.runtime-optimization-contract/v1';
  contractId: string;
  kind: RuntimeOptimizationCandidateKind;
  campaign: {
    owner: string;
    changeClass:
      | 'scheduling-allocation-cache'
      | 'numerical-kernel'
      | 'precision-quantization'
      | 'model-artifact'
      | 'adapter'
      | 'provider-integration';
    causalHypothesis: string;
    expectedMetric: {
      path: RuntimeOptimizationContract['measurement']['metricPath'];
      direction: RuntimeOptimizationDirection;
      minImprovementPercent: number;
    };
    controlMetric: {
      path: RuntimeOptimizationContract['verification']['comparisons'][number]['path'];
      expectation: 'unchanged';
    };
    endToEndAcceptanceMetric: {
      path: RuntimeOptimizationContract['measurement']['metricPath'];
      direction: RuntimeOptimizationDirection;
      minImprovementPercent: number;
    };
    budgets: {
      maxCandidates: number;
      maxCommandRunsPerCandidate: number;
    };
    stoppingRule: {
      kind: 'fixed-contract' | 'bonferroni-fixed-looks';
      retainNegativeResults: true;
    };
    retryConditions: string[];
    revocationConditions: string[];
  };
  model: {
    modelId: string;
    modelUrl: string | null;
    expectedExecutionContractHash: `sha256:${string}` | null;
  };
  baseline: {
    runtimeProfile: null;
    runtimeConfig: Record<string, unknown>;
  };
  workload: {
    type: RuntimeOptimizationWorkload;
    request: {
      inferenceInput?: Record<string, unknown> | null;
      cacheMode?: 'cold' | 'warm' | null;
      loadMode?: 'opfs' | 'http' | 'memory' | 'file' | null;
    };
  };
  mutationPolicy:
    | {
      dimensions: Array<{ path: string; values: unknown[] }>;
      maxCandidates: number;
    }
    | {
      references: Array<{ registryId: string; digest: `sha256:${string}` }>;
      maxCandidates: number;
    };
  verification: {
    comparisons: Array<{
      path: 'result.output'
        | 'result.metrics.referenceTranscript.tokens.generatedTokenIdsHash'
        | 'result.metrics.referenceTranscript.output.textHash';
      mode: 'canonical_exact';
    }>;
  };
  measurement: {
    metricPath: 'result.metrics.decodeTokensPerSec'
      | 'result.metrics.embeddingMs'
      | 'result.metrics.rerankMs'
      | 'result.timing.decodeTokensPerSec'
      | 'result.timing.totalRunMs';
    direction: RuntimeOptimizationDirection;
    pairCount: number;
    minValidPairs: number;
    minImprovementPercent: number;
    requirePositiveConfidence: boolean;
    maxRelativeStdDevPercent: number | null;
    orderPolicy?: {
      kind: 'randomized-blocks';
      seed: number;
      blockSize: 2;
    };
    sequentialDecision?: {
      kind: 'bonferroni-fixed-looks';
      lookEveryPairs: number;
      minimumPairs: number;
      maximumLooks: number;
      alpha: number;
    };
  };
  neighboringWorkloads?: Array<{
    guardId: string;
    workload: RuntimeOptimizationContract['workload'];
    metricPath: RuntimeOptimizationContract['measurement']['metricPath'];
    direction: RuntimeOptimizationDirection;
    maxRegressionPercent: number;
    pairCount: number;
  }>;
}

export interface RuntimeOptimizationCandidate {
  schema: 'doppler.runtime-optimization-candidate/v1';
  candidateId: string;
  contractHash: `sha256:${string}`;
  parentHash: `sha256:${string}`;
  kind: Exclude<RuntimeOptimizationCandidateKind, 'runtime_profile'>;
  patch: Array<{ op: 'set'; path: string; value: unknown }>;
  registeredReference?: { registryId: string; digest: `sha256:${string}` };
}

export interface RuntimeOptimizationReceipt {
  schema: 'doppler.runtime-optimization-receipt/v1';
  contractId: string;
  contractHash: `sha256:${string}`;
  candidateId: string;
  candidateHash: `sha256:${string}`;
  parentHash: `sha256:${string}`;
  candidateKind: string;
  registeredReference: { registryId: string; digest: `sha256:${string}` } | null;
  campaign: RuntimeOptimizationContract['campaign'];
  model: RuntimeOptimizationContract['model'];
  runtimeInputs: Record<string, unknown>;
  verification: Record<string, unknown>;
  measurement: Record<string, unknown>;
  neighboringWorkloadGuards: Record<string, unknown> | null;
  decision: {
    accepted: boolean;
    status: 'accepted' | 'rejected' | 'invalid';
    reasons: string[];
  };
  promotion: {
    authority: 'human';
    recommended: boolean;
    runtimeMutationApplied: false;
    requiredStages: ['shadow', 'canary'];
    revocationConditions: string[];
  };
  receiptHash: `sha256:${string}`;
}

export interface RuntimeOptimizationEvaluationOptions {
  runCommand?: (request: Record<string, unknown>, options?: Record<string, unknown>) => Promise<Record<string, unknown>>;
  commandOptions?: Record<string, unknown>;
  candidateRegistry?: Record<string, unknown>;
  signal?: AbortSignal | null;
  onEvent?: (event: Record<string, unknown>) => void;
}

export declare const RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA: 'doppler.runtime-optimization-contract/v1';
export declare const RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA: 'doppler.runtime-optimization-candidate/v1';
export declare const RUNTIME_OPTIMIZATION_RECEIPT_SCHEMA: 'doppler.runtime-optimization-receipt/v1';
export declare const RUNTIME_OPTIMIZATION_CANDIDATE_REGISTRY_SCHEMA:
  'doppler.runtime-optimization-candidate-registry/v1';

export declare function validateRuntimeOptimizationContract(
  input: RuntimeOptimizationContract
): RuntimeOptimizationContract;
export declare function hashRuntimeOptimizationContract(input: RuntimeOptimizationContract): `sha256:${string}`;
export declare function enumerateRuntimeOptimizationCandidates(
  input: RuntimeOptimizationContract
): RuntimeOptimizationCandidate[];
export declare function validateRuntimeOptimizationCandidate(
  candidate: RuntimeOptimizationCandidate,
  contract: RuntimeOptimizationContract
): RuntimeOptimizationCandidate;
export declare function materializeRuntimeOptimizationCandidate(
  contract: RuntimeOptimizationContract,
  candidate: RuntimeOptimizationCandidate,
  options?: { candidateRegistry?: Record<string, unknown> }
): { runtimeProfile: null; runtimeConfig: Record<string, unknown> };
export declare function validateRuntimeOptimizationCandidateRegistry(
  registry: Record<string, unknown>
): Record<string, unknown>;
export declare function evaluateBrowserRuntimeOptimizationCandidate(
  contract: RuntimeOptimizationContract,
  candidate: RuntimeOptimizationCandidate,
  options?: RuntimeOptimizationEvaluationOptions
): Promise<RuntimeOptimizationReceipt>;
