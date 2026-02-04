/**
 * Energy Pipeline Config Schema
 *
 * Defaults and runtime config for energy-based inference loops.
 *
 * @module config/schema/energy
 */

export interface EnergyStateConfigSchema {
  width: number;
  height: number;
  channels: number;
  dtype: string;
}

export interface EnergyInitConfigSchema {
  mode: 'normal' | 'uniform' | 'zeros';
  seed: number;
  scale: number;
}

export interface EnergyTargetConfigSchema {
  mode: 'normal' | 'uniform' | 'zeros';
  seed: number;
  scale: number;
}

export interface EnergyLoopConfigSchema {
  maxSteps: number;
  minSteps: number;
  stepSize: number;
  gradientScale: number;
  convergenceThreshold: number;
}

export interface EnergyDiagnosticsConfigSchema {
  readbackEvery: number;
  traceEvery: number;
  historyLimit: number;
}

export interface EnergyQuintelRulesConfigSchema {
  mirrorX: boolean;
  mirrorY: boolean;
  diagonal: boolean;
  count: boolean;
  center: boolean;
}

export interface EnergyQuintelWeightsConfigSchema {
  symmetry: number;
  count: number;
  center: number;
  binarize: number;
}

export interface EnergyQuintelClampConfigSchema {
  min: number;
  max: number;
}

export interface EnergyQuintelConfigSchema {
  size: number;
  rules: EnergyQuintelRulesConfigSchema;
  weights: EnergyQuintelWeightsConfigSchema;
  clamp: EnergyQuintelClampConfigSchema;
  countTarget: number;
  centerTarget: number;
}

export interface EnergyConfigSchema {
  problem: 'l2' | 'quintel';
  state: EnergyStateConfigSchema;
  init: EnergyInitConfigSchema;
  target: EnergyTargetConfigSchema;
  loop: EnergyLoopConfigSchema;
  diagnostics: EnergyDiagnosticsConfigSchema;
  quintel: EnergyQuintelConfigSchema;
}

export interface EnergyModelConfigSchema {
  problem?: 'l2' | 'quintel';
  state?: Partial<EnergyStateConfigSchema>;
  init?: Partial<EnergyInitConfigSchema>;
  target?: Partial<EnergyTargetConfigSchema>;
  loop?: Partial<EnergyLoopConfigSchema>;
  diagnostics?: Partial<EnergyDiagnosticsConfigSchema>;
  quintel?: Partial<EnergyQuintelConfigSchema>;
  shape?: number[];
}

export declare const DEFAULT_ENERGY_STATE_CONFIG: EnergyStateConfigSchema;
export declare const DEFAULT_ENERGY_INIT_CONFIG: EnergyInitConfigSchema;
export declare const DEFAULT_ENERGY_TARGET_CONFIG: EnergyTargetConfigSchema;
export declare const DEFAULT_ENERGY_LOOP_CONFIG: EnergyLoopConfigSchema;
export declare const DEFAULT_ENERGY_DIAGNOSTICS_CONFIG: EnergyDiagnosticsConfigSchema;
export declare const DEFAULT_ENERGY_QUINTEL_RULES: EnergyQuintelRulesConfigSchema;
export declare const DEFAULT_ENERGY_QUINTEL_WEIGHTS: EnergyQuintelWeightsConfigSchema;
export declare const DEFAULT_ENERGY_QUINTEL_CLAMP: EnergyQuintelClampConfigSchema;
export declare const DEFAULT_ENERGY_QUINTEL_CONFIG: EnergyQuintelConfigSchema;
export declare const DEFAULT_ENERGY_CONFIG: EnergyConfigSchema;
