/**
 * Quintel energy loop utilities.
 *
 * @module inference/energy/quintel
 */

export interface QuintelRuleConfig {
  mirrorX: boolean;
  mirrorY: boolean;
  diagonal: boolean;
  count: boolean;
  center: boolean;
}

export interface QuintelWeightConfig {
  symmetry: number;
  count: number;
  center: number;
  binarize: number;
}

export interface QuintelClampConfig {
  min: number;
  max: number;
}

export interface QuintelEnergyConfig {
  size: number;
  rules: QuintelRuleConfig;
  weights: QuintelWeightConfig;
  clamp: QuintelClampConfig;
  countTarget: number;
  centerTarget: number;
}

export interface QuintelEnergyComponents {
  symmetry: number | null;
  count: number | null;
  center: number | null;
  binarize: number | null;
}

export interface QuintelEnergyLoopConfig {
  maxSteps: number;
  minSteps: number;
  stepSize: number;
  gradientScale: number;
  convergenceThreshold: number | null;
}

export interface QuintelEnergyDiagnosticsConfig {
  readbackEvery: number;
  historyLimit: number;
  traceEvery: number;
}

export interface QuintelEnergyLoopOptions {
  state: Float32Array;
  size: number;
  config: QuintelEnergyConfig;
  loop: QuintelEnergyLoopConfig;
  diagnostics: QuintelEnergyDiagnosticsConfig;
  onProgress?: (payload: {
    stage: 'energy';
    percent: number;
    message: string;
  }) => void;
  onTrace?: (step: number, energy: number, components: QuintelEnergyComponents) => void;
  traceEvery?: number;
}

export interface QuintelEnergyLoopResult {
  state: Float32Array;
  energy: number | null;
  energyHistory: number[];
  energyComponents: QuintelEnergyComponents | null;
  stepTimesMs: number[];
  steps: number;
  totalTimeMs: number;
}

export function mergeQuintelConfig(
  base: QuintelEnergyConfig,
  override?: Partial<QuintelEnergyConfig> | null
): QuintelEnergyConfig;

export function runQuintelEnergyLoop(options: QuintelEnergyLoopOptions): QuintelEnergyLoopResult;
