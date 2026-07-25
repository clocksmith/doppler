export interface RuntimeCompositionBridge {
  getRuntimeConfig: () => Record<string, unknown> | null;
  setRuntimeConfig: (runtimeConfig: Record<string, unknown> | null) => void;
}

export interface RuntimeInputDocument {
  readonly kind: 'configChain' | 'runtimeProfile' | 'runtimeConfigUrl' | 'runtimeConfig';
  readonly ref: string;
  readonly config: Record<string, unknown>;
  readonly runtime: Record<string, unknown>;
}

export interface RuntimeInputCompositionHandlers {
  loadRuntimeConfigFromRef?: (
    ref: string,
    options?: Record<string, unknown>
  ) => Promise<Record<string, unknown> | {
    config: Record<string, unknown>;
    runtime: Record<string, unknown>;
  } | null>;
  loadRuntimeProfile?: (
    runtimeProfile: string,
    options?: Record<string, unknown>
  ) => Promise<{
    config: Record<string, unknown>;
    runtime: Record<string, unknown>;
  }>;
  loadRuntimeConfigFromUrl?: (
    runtimeConfigUrl: string,
    options?: Record<string, unknown>
  ) => Promise<{
    config: Record<string, unknown>;
    runtime: Record<string, unknown>;
  }>;
}

export interface OrderedRuntimeInputs {
  configChain?: string[] | null;
  runtimeProfile?: string | null;
  runtimeConfigUrl?: string | null;
  runtimeConfig?: Record<string, unknown> | null;
}

export interface ResolvedRuntimeInputs {
  runtime: Record<string, unknown> | null;
  documents: ReadonlyArray<RuntimeInputDocument>;
}

export declare function resolveRuntimeFromConfig(
  config: Record<string, unknown> | null | undefined
): Record<string, unknown> | null;

export declare function resolveOrderedRuntimeInputs(
  initialRuntime: Record<string, unknown> | null,
  inputs?: OrderedRuntimeInputs,
  handlers?: RuntimeInputCompositionHandlers,
  options?: Record<string, unknown>
): Promise<ResolvedRuntimeInputs>;

export declare function applyOrderedRuntimeInputs(
  runtimeBridge: RuntimeCompositionBridge,
  inputs?: OrderedRuntimeInputs,
  handlers?: RuntimeInputCompositionHandlers,
  options?: Record<string, unknown>
): Promise<ResolvedRuntimeInputs>;
