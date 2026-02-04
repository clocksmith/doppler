/**
 * Energy Pipeline Types
 *
 * @module inference/energy/types
 */

import type { ArrayStats } from '../../debug/stats.js';
import type { EnergyQuintelConfigSchema } from '../../config/schema/energy.schema.js';

export type EnergyProblem = 'l2' | 'quintel' | 'vliw';

export interface EnergyComponents {
  symmetry?: number | null;
  count?: number | null;
  center?: number | null;
  binarize?: number | null;
}

export interface EnergyRequest {
  problem?: EnergyProblem;
  quintel?: Partial<EnergyQuintelConfigSchema>;
  vliw?: {
    tasks?: Array<{
      id: number;
      engine: string;
      reads?: number[];
      writes?: number[];
      deps?: number[];
      bundle?: number;
    }>;
    caps?: Record<string, number>;
    search?: {
      restarts?: number;
      temperatureStart?: number;
      temperatureDecay?: number;
      mutationCount?: number;
    };
  };
  shape?: number[];
  width?: number;
  height?: number;
  channels?: number;
  steps?: number;
  stepSize?: number;
  gradientScale?: number;
  convergenceThreshold?: number;
  seed?: number;
  targetSeed?: number;
  initMode?: 'normal' | 'uniform' | 'zeros';
  targetMode?: 'normal' | 'uniform' | 'zeros';
  initScale?: number;
  targetScale?: number;
  readbackEvery?: number;
}

export interface EnergyResult {
  shape: number[];
  dtype: string;
  steps: number;
  energy?: number | null;
  energyComponents?: EnergyComponents | null;
  state: Float32Array;
  energyHistory: number[];
  stateStats: ArrayStats;
  totalTimeMs: number;
  metrics?: {
    cycles: number;
    utilization: number;
    violations: number;
  };
  problem?: EnergyProblem;
}

export interface EnergyStats {
  totalTimeMs?: number;
  steps?: number;
  stepTimesMs?: number[];
  energyHistory?: number[];
  readbackCount?: number;
  energy?: number | null;
  energyComponents?: EnergyComponents | null;
  stateStats?: ArrayStats;
}
