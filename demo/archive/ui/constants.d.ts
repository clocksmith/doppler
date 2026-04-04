export interface EnergyDemo {
  id: string;
  problem: string;
  label: string;
  description: string;
  defaults: Record<string, unknown>;
}

export interface DiagnosticsSuiteDefaults {
  suite: string;
  runtimeProfile: string;
}

export interface DiagnosticsSuiteInfo {
  description: string;
  requiresModel: boolean;
  requiresBenchIntent: boolean;
}

export interface RuntimeProfileEntry {
  id: string;
  label: string;
  base: boolean;
  override: boolean;
}

export declare const ENERGY_DEMOS: readonly EnergyDemo[];
export declare const DEFAULT_ENERGY_DEMO_ID: string;
export declare const ENERGY_METRIC_LABELS: Record<string, Record<string, string>>;
export declare const RUNTIME_PROFILE_REGISTRY: readonly RuntimeProfileEntry[];
export declare const DIAGNOSTICS_SUITE_INFO: Record<string, DiagnosticsSuiteInfo>;
export declare const DIAGNOSTICS_SUITE_ORDER: readonly string[];
export declare const BENCH_INTENTS: ReadonlySet<string>;
export declare const DEFAULT_RUNTIME_PROFILE: string;
export declare const DIAGNOSTICS_DEFAULTS: Record<string, DiagnosticsSuiteDefaults>;
