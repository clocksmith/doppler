export declare const WGSL_REPAIR_TASK_CONTRACT: 'replacement_only_wgsl_span_v1';
export declare const WGSL_REPAIR_MUTATION_OPERATORS: readonly string[];
export declare const VERIFIER_GUIDED_ARTIFACT_TYPES: readonly string[];

export interface WgslRepairMutation {
  operator: string;
  detail: string | null;
  spanStart: number;
  spanEnd: number;
  originalSpan: string;
  mutatedSpan: string;
  mutatedSource: string;
}

export interface WgslRepairSourceRecord {
  sourceId: string;
  sourcePath: string;
  revision: string;
  license: string;
  source: string;
  kernelFamilyId?: string;
}

export interface WgslRepairTask {
  schemaVersion: 1;
  taskContract: string;
  id: string;
  rowId: string;
  taskId: string;
  kernelFamilyId: string;
  sourceId: string;
  sourcePath: string;
  sourceRevision: string;
  sourceLicense: string;
  sourceSha256: string;
  mutation: Record<string, unknown>;
  span: { start: number; end: number; broken: string; reference: string };
  prompt: string;
  completion: string;
  source: string;
  mutatedSource: string;
}

export declare function deriveKernelFamily(sourceId: string, sourcePath: string): string;
export declare function createWgslRepairMutations(
  source: string,
  operators?: readonly string[]
): WgslRepairMutation[];
export declare function buildWgslRepairTask(
  sourceRecord: WgslRepairSourceRecord,
  mutation: WgslRepairMutation
): WgslRepairTask;
export declare function parseReplacementOnlyResponse(response: unknown): {
  ok: boolean;
  replacement: string;
  violations: string[];
};
export declare function applyWgslRepairResponse(task: WgslRepairTask, response: unknown): {
  ok: boolean;
  replacement: string;
  violations: string[];
  candidateSource: string;
  candidateSha256: string;
};
export declare function buildWgslRewardVector(input: Record<string, unknown>): Record<string, unknown>;
export declare function computeGroupRelativeAdvantages(rewards: number[], epsilon?: number): {
  mean: number;
  variance: number;
  standardDeviation: number;
  epsilon: number;
  zeroVariance: boolean;
  zeroVariancePolicy: 'zero_advantages';
  advantages: number[];
};
export declare function buildTrainingRolloutGroup(input: Record<string, unknown>): Record<string, unknown>;
export declare function selectRejectionSamples(groups: Array<Record<string, unknown>>): Array<Record<string, unknown>>;
export declare function deriveDpoPreferencePairs(
  groups: Array<Record<string, unknown>>,
  options?: { minimumRewardGap?: number }
): Array<Record<string, unknown>>;
export declare function buildTrainingPromotionDecision(input: Record<string, unknown>): Record<string, unknown>;
export declare function validateVerifierGuidedArtifact<T extends Record<string, unknown>>(artifact: T): T;
export declare function hashVerifierGuidedArtifact(artifact: Record<string, unknown>): string;
