export interface ResolvedActivationDtype {
  activationDtype: 'f16' | 'f32' | null;
  source: 'executionPlan' | 'runtimeConfig' | 'modelConfig' | 'none';
  allSources: {
    executionPlan: 'f16' | 'f32' | null;
    runtimeConfig: 'f16' | 'f32' | null;
    modelConfig: 'f16' | 'f32' | null;
  };
}

export interface DtypeConsistencyResult {
  consistent: boolean;
  values: {
    executionPlan: 'f16' | 'f32' | null;
    runtimeConfig: 'f16' | 'f32' | null;
    layerContext: 'f16' | 'f32' | null;
  };
}

/**
 * Resolve activation dtype from all available sources.
 * Returns { activationDtype, source, allSources } for diagnostics.
 */
export declare function resolveActivationDtype(
  executionPlanState: Record<string, unknown> | null,
  runtimeConfig: Record<string, unknown> | null,
  modelConfig: Record<string, unknown> | null
): ResolvedActivationDtype;

/**
 * Assert dtype consistency across all resolution paths.
 * Logs a warning if they disagree. Does not throw.
 */
export declare function assertDtypeConsistency(
  executionPlanState: Record<string, unknown> | null,
  runtimeConfig: Record<string, unknown> | null,
  layerContext: Record<string, unknown> | null
): DtypeConsistencyResult;
