export interface KernelPathBuilderRuntimeFact {
  id: string;
  kind: string;
  label: string;
  summary: string;
  affectedLayers?: number[];
  assumptions: Record<string, unknown>;
  sourceRefs: string[];
}

export interface BuildCustomRuntimeFactsOptions {
  modelId?: string | null;
  manifestInference?: Record<string, unknown> | null;
}

export declare function buildCustomRuntimeFacts(
  options?: BuildCustomRuntimeFactsOptions
): KernelPathBuilderRuntimeFact[];
