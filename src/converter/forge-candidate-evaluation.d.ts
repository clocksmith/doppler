export type ForgeEvaluationJson = null | boolean | number | string
  | ForgeEvaluationJson[] | { [key: string]: ForgeEvaluationJson };

export const FORGE_EVALUATION_SCHEMA: 'doppler.forge-candidate-evaluation/v1';

export interface ForgeEvaluationContract {
  schema: typeof FORGE_EVALUATION_SCHEMA;
  evaluationId: string;
  modelIRHash: string;
  candidateHashes: string[];
  referenceHash: string;
  scope: {
    surface: string;
    runtimeHash: string;
    environmentHash: string;
    workloadHash: string;
    cacheMode: string;
    loadMode: string;
  };
  sampling: { warmupRuns: number; timedRuns: number; order: 'balanced-rotation'; seed: number };
  checks: Array<{ id: string } & (
    { mode: 'canonical-exact'; maxAbsoluteError: null }
    | { mode: 'absolute-array'; maxAbsoluteError: number }
  )>;
  metrics: Array<{ id: string; unit: string; direction: 'minimize' | 'maximize'; limit: number | null }>;
  selection: 'observed-range-pareto';
}

export interface ForgeEvaluationReference {
  schema: 'doppler.forge-source-reference/v1';
  sourceHash: string;
  oracleHash: string;
  cases: Array<{ id: string; input: ForgeEvaluationJson; expected: Record<string, ForgeEvaluationJson> }>;
}

export interface ForgeEvaluationAttempt {
  attemptId: string;
  candidateHash: string;
  phase: 'warmup' | 'timed';
  run: number;
  caseId: string;
  inputHash: string;
}

export interface ForgeEvaluationObservation {
  attemptId: string;
  candidateHash: string;
  contractHash: string;
  scopeHash: string;
  inputHash: string;
  status: 'completed' | 'failed' | 'cancelled';
  output: Record<string, ForgeEvaluationJson> | null;
  metrics: Record<string, number> | null;
  error: string | null;
}

export interface ForgeEvaluationInput {
  contract: ForgeEvaluationContract;
  reference: ForgeEvaluationReference;
  observations: ForgeEvaluationObservation[];
}

export interface ForgeEvaluationReceipt {
  schema: 'doppler.forge-candidate-evaluation-receipt/v1';
  contractHash: string;
  referenceHash: string;
  observationsHash: string;
  modelIRHash: string;
  selection: ForgeEvaluationContract['selection'];
  scope: ForgeEvaluationContract['scope'];
  attempts: Array<ForgeEvaluationAttempt & {
    passed: boolean;
    reason: string | null;
    error?: string;
    checks: Array<{ id: string; passed: boolean; reason?: string; maxAbsoluteError?: number }>;
  }>;
  candidates: Array<{
    candidateHash: string;
    eligible: boolean;
    retained: boolean;
    failedAttemptIds: string[];
    dominatedBy: string[];
    ranges: Record<string, Record<string, { min: number; max: number; count: number }>>;
  }>;
  selectedCandidateHashes: string[];
  claimAllowed: false;
  promotionAllowed: false;
}

export declare function validateForgeEvaluationContract(
  contract: ForgeEvaluationContract, reference: ForgeEvaluationReference
): ForgeEvaluationContract;
export declare function createForgeEvaluationSchedule(
  contract: ForgeEvaluationContract, reference: ForgeEvaluationReference
): ForgeEvaluationAttempt[];
export declare function evaluateForgeCandidates(input: ForgeEvaluationInput): ForgeEvaluationReceipt;
export declare function runForgeCandidateEvaluation(options: {
  contract: ForgeEvaluationContract;
  reference: ForgeEvaluationReference;
  runAttempt(context: {
    attempt: ForgeEvaluationAttempt;
    input: ForgeEvaluationJson;
    contractHash: string;
    scopeHash: string;
    signal?: AbortSignal;
  }): Promise<ForgeEvaluationObservation>;
  signal?: AbortSignal;
  onObservation?(observation: ForgeEvaluationObservation): void | Promise<void>;
}): Promise<ForgeEvaluationInput & { receipt: ForgeEvaluationReceipt }>;
