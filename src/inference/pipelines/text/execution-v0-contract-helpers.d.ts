export declare function cloneJson<T>(value: T): T;
export declare function validateManifestSessionDefaultsContract(manifestInference: Record<string, unknown> | null): void;
export declare function isPhaseMatch(phase: string, targetPhase: string): boolean;
export declare function stepHasLayer(step: Record<string, unknown>, layerIdx: number): boolean;
export declare function normalizePhase(value: unknown, label: string): string;
export declare function normalizeSection(value: unknown, label: string): string;
export declare function normalizeSlot(value: unknown, label: string): string;
export declare function createSourceTrace(): { session: Record<string, unknown>; steps: Record<string, unknown> };
export declare function setSourceTrace(trace: Record<string, unknown>, path: string, source: string): void;
export declare function collectLeafPaths(value: unknown, prefix?: string[], out?: string[][]): string[][];
export declare function hasDefinedPath(root: unknown, pathSegments: string[]): boolean;
export declare function validateStepShape(step: Record<string, unknown>, index: number): void;
export declare function assertExecutionRuntimeOverlay(runtimeInference: Record<string, unknown> | null | undefined): void;
export declare function validateUniqueStepIds(steps: Array<Record<string, unknown>>): void;
export declare function hasExecutionV0(manifestInference: Record<string, unknown> | null | undefined): boolean;
export declare function assertExecutionV0Schema(manifestInference: Record<string, unknown> | null | undefined): void;
export declare function applyExecutionPatchAtomic(
  baseSteps: Array<Record<string, unknown>>,
  patch: Record<string, unknown> | null | undefined
): Array<Record<string, unknown>>;
export declare function indexRuntimePatchMeta(
  patch: Record<string, unknown> | null | undefined
): {
  addedSteps: Set<string>;
  precisionFieldsByStep: Map<string, Set<string>>;
  kvIOFieldsByStep: Set<string>;
};
export declare function requireSessionActivationDtype(
  sessionDefaults: Record<string, unknown> | null | undefined,
  label?: string
): string;
export declare function createInitialSlotDtypes(sessionDefaults: Record<string, unknown>): Map<string, string>;
export declare function resolvePhaseSteps(
  phase: string,
  steps: Array<Record<string, unknown>>,
  sessionDefaults: Record<string, unknown>,
  profileIndex: Map<string, unknown>,
  policies: Record<string, unknown>,
  options?: Record<string, unknown>
): {
  steps: Array<Record<string, unknown>>;
  finalSlotDtypes: Map<string, string>;
};
export declare function normalizeRuntimeSessionForExecutionV0(
  runtimeSession: Record<string, unknown> | null | undefined,
  manifestInference: Record<string, unknown> | null | undefined,
  defaultComputeDefaults: Record<string, unknown>
): Record<string, unknown> | null | undefined;
export declare function validatePhaseBoundaryCompatibility(options: Record<string, unknown>): void;
export declare function assertKVLayoutExecutionCompatibility(
  steps: Array<Record<string, unknown>>,
  sessionDefaults: Record<string, unknown>
): void;
export declare const buildKernelProfileKey: (
  kernelRef: Record<string, unknown> | null | undefined,
  step?: Record<string, unknown> | null | undefined
) => string;
export declare const indexKernelProfiles: (sessionDefaults: Record<string, unknown>) => Map<string, unknown>;
export declare const normalizeDtype: (value: unknown, label: string) => string;
