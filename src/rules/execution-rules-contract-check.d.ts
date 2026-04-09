export interface InferenceExecutionRulesContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: Array<{ id: string; ok: boolean }>;
  errors: string[];
  stats: {
    decodeRecorderRules: number;
    profileDecodeRecorderRules: number;
    batchDecodeRules: number;
    maxBatchDecodeTokenRules: number;
    decodeRecorderContexts: number;
    profileDecodeRecorderContexts: number;
    batchDecodeContexts: number;
    maxBatchDecodeTokenContexts: number;
  };
}

export declare function buildInferenceExecutionRulesContractArtifact(
  ruleGroup: Record<string, unknown> | null | undefined
): InferenceExecutionRulesContractArtifact;
