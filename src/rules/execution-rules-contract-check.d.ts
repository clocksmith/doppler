export interface InferenceExecutionRulesContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: Array<{ id: string; ok: boolean }>;
  errors: string[];
  stats: {
    decodeRecorderRules: number;
    batchDecodeRules: number;
    decodeRecorderContexts: number;
    batchDecodeContexts: number;
  };
}

export declare function buildInferenceExecutionRulesContractArtifact(
  ruleGroup: Record<string, unknown> | null | undefined
): InferenceExecutionRulesContractArtifact;
