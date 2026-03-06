export interface LayerPatternContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: Array<{ id: string; ok: boolean }>;
  errors: string[];
  stats: {
    patternKindRules: number;
    layerTypeRules: number;
    patternKindContexts: number;
    layerTypeContexts: number;
  };
}

export declare function buildLayerPatternContractArtifact(
  ruleGroup: Record<string, unknown> | null | undefined
): LayerPatternContractArtifact;
