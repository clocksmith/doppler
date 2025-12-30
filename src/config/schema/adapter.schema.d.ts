export type LoraTargetModule =
  | 'q_proj'
  | 'k_proj'
  | 'v_proj'
  | 'o_proj'
  | 'gate_proj'
  | 'up_proj'
  | 'down_proj'
  | 'gate_up_proj';

export const VALID_LORA_TARGET_MODULES: readonly LoraTargetModule[];

export interface AdapterValidationConfig {
  minRank: number;
  maxRank: number;
  minAlpha: number;
  maxNameLength: number;
  maxDescriptionLength: number;
}

export const DEFAULT_ADAPTER_VALIDATION_CONFIG: AdapterValidationConfig;

export type StackingStrategy = 'sum' | 'concat' | 'weighted';

export interface AdapterStackConfig {
  strategy: StackingStrategy;
  normalizeWeights: boolean;
  minWeight: number;
  maxWeight: number;
}

export const DEFAULT_ADAPTER_STACK_CONFIG: AdapterStackConfig;

export type SortOrder = 'asc' | 'desc';

export interface AdapterRegistryConfig {
  dbName: string;
  dbVersion: number;
  storeName: string;
  defaultSortBy: string;
  defaultSortOrder: SortOrder;
}

export const DEFAULT_ADAPTER_REGISTRY_CONFIG: AdapterRegistryConfig;

export interface AdapterConfig {
  validation: AdapterValidationConfig;
  stacking: AdapterStackConfig;
  registry: AdapterRegistryConfig;
  targetModules: readonly LoraTargetModule[];
}

export const DEFAULT_ADAPTER_CONFIG: AdapterConfig;
