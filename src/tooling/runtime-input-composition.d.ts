export interface RuntimeCompositionBridge {
  getRuntimeConfig: () => Record<string, unknown> | null;
  setRuntimeConfig: (runtimeConfig: Record<string, unknown> | null) => void;
}

export interface RuntimeInputCompositionHandlers {
  loadRuntimeConfigFromRef?: (
    ref: string,
    options?: Record<string, unknown>
  ) => Promise<Record<string, unknown> | null>;
  applyRuntimePreset?: (
    runtimePreset: string,
    options?: Record<string, unknown>
  ) => Promise<void>;
  applyRuntimeConfigFromUrl?: (
    runtimeConfigUrl: string,
    options?: Record<string, unknown>
  ) => Promise<void>;
}

export interface OrderedRuntimeInputs {
  configChain?: string[] | null;
  runtimePreset?: string | null;
  runtimeConfigUrl?: string | null;
  runtimeConfig?: Record<string, unknown> | null;
  runtimeContractPatch?: Record<string, unknown> | null | (() => Record<string, unknown> | null);
}

export declare function resolveRuntimeFromConfig(
  config: Record<string, unknown> | null | undefined
): Record<string, unknown> | null;

export declare function applyOrderedRuntimeInputs(
  runtimeBridge: RuntimeCompositionBridge,
  inputs?: OrderedRuntimeInputs,
  handlers?: RuntimeInputCompositionHandlers,
  options?: Record<string, unknown>
): Promise<void>;
