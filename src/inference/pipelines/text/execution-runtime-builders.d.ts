export declare const PIPELINE_COMPATIBLE_OPS: ReadonlySet<string>;

export declare function normalizeDtype(value: unknown, label: string): 'f16' | 'f32';
export declare function isPhaseMatch(phase: string, targetPhase: string): boolean;
export declare function stepHasLayer(step: { layers: 'all' | number[] }, layerIdx: number): boolean;
export declare function requireSessionActivationDtype(
  sessionDefaults: Record<string, unknown> | null | undefined,
  label?: string
): 'f16' | 'f32';

export declare function resolveFinitenessFallbackKernelPathId(
  kernelPathId: string | null | undefined
): string | null;

export declare function buildInlineKernelPath(
  steps: readonly Record<string, unknown>[],
  sessionDefaults: Record<string, unknown> | null,
  modelId: string,
  numLayers: number,
  finitenessFallbackKernelPathId?: string | null
): Record<string, unknown> | null;

export declare function assertKernelPathSessionCompatibility(
  path: Record<string, unknown> | null | undefined,
  sessionDefaults: Record<string, unknown> | null | undefined
): void;

export interface BuildLayerPipelineOptions {
  /** When true, throws on incompatible ops instead of returning a degraded result. */
  strict?: boolean;
}

export declare function buildLayerPipelineFromExecution(
  steps: readonly Record<string, unknown>[],
  options?: BuildLayerPipelineOptions
): { steps: Record<string, unknown>[]; overrides: unknown[]; hasIncompatibleOps: false }
  | { incompatibleOps: string[]; hasIncompatibleOps: true }
  | null;

export declare function buildSessionRuntimePatch(
  sessionDefaults: Record<string, unknown> | null | undefined,
  options?: {
    includeDecodeLoop?: boolean;
  }
): Record<string, unknown>;
