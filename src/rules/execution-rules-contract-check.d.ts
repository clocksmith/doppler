import type { RuleFileStructureArtifact } from './rule-matcher.js';

export type InferenceExecutionRulesContractArtifact = RuleFileStructureArtifact;

export declare function buildInferenceExecutionRulesContractArtifact(
  ruleGroup: Record<string, unknown> | null | undefined
): InferenceExecutionRulesContractArtifact;
