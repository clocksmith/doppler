export type RuntimeOptimizationWorkload = 'inference' | 'embedding' | 'rerank';
export type RuntimeOptimizationDirection = 'maximize' | 'minimize';

export interface RuntimeOptimizationContract {
  schema: 'doppler.runtime-optimization-contract/v1';
  contractId: string;
  kind: 'runtime_profile';
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
  mutationPolicy: {
    dimensions: Array<{ path: string; values: unknown[] }>;
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
  };
}

export interface RuntimeOptimizationCandidate {
  schema: 'doppler.runtime-optimization-candidate/v1';
  candidateId: string;
  contractHash: `sha256:${string}`;
  parentHash: `sha256:${string}`;
  patch: Array<{ op: 'set'; path: string; value: unknown }>;
}

export interface RuntimeOptimizationReceipt {
  schema: 'doppler.runtime-optimization-receipt/v1';
  contractId: string;
  contractHash: `sha256:${string}`;
  candidateId: string;
  candidateHash: `sha256:${string}`;
  parentHash: `sha256:${string}`;
  model: RuntimeOptimizationContract['model'];
  runtimeInputs: Record<string, unknown>;
  verification: Record<string, unknown>;
  measurement: Record<string, unknown>;
  decision: {
    accepted: boolean;
    status: 'accepted' | 'rejected' | 'invalid';
    reasons: string[];
  };
  receiptHash: `sha256:${string}`;
}

export interface RuntimeOptimizationEvaluationOptions {
  runCommand?: (request: Record<string, unknown>, options?: Record<string, unknown>) => Promise<Record<string, unknown>>;
  commandOptions?: Record<string, unknown>;
  signal?: AbortSignal | null;
  onEvent?: (event: Record<string, unknown>) => void;
}

export declare const RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA: 'doppler.runtime-optimization-contract/v1';
export declare const RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA: 'doppler.runtime-optimization-candidate/v1';
export declare const RUNTIME_OPTIMIZATION_RECEIPT_SCHEMA: 'doppler.runtime-optimization-receipt/v1';

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
  candidate: RuntimeOptimizationCandidate
): { runtimeProfile: null; runtimeConfig: Record<string, unknown> };
export declare function evaluateBrowserRuntimeOptimizationCandidate(
  contract: RuntimeOptimizationContract,
  candidate: RuntimeOptimizationCandidate,
  options?: RuntimeOptimizationEvaluationOptions
): Promise<RuntimeOptimizationReceipt>;
