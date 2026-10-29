import type { RuleFileStructureArtifact } from './rule-matcher.js';

export type LayerPatternContractArtifact = RuleFileStructureArtifact;

export declare function buildLayerPatternContractArtifact(
  ruleGroup: Record<string, unknown> | null | undefined
): LayerPatternContractArtifact;
