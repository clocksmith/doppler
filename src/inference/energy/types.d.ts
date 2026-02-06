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
    dependencyModel?: {
      includes_raw: boolean;
      includes_waw: boolean;
      includes_war: boolean;
      temp_hazard_tags: boolean;
      read_after_read: boolean;
      latency?: {
        default: number;
        raw?: number;
        waw?: number;
        war?: number;
        temp?: number;
        rar?: number;
      };
    };
    search?: {
      restarts?: number;
      temperatureStart?: number;
      temperatureDecay?: number;
      mutationCount?: number;
      policy?: 'weights' | 'priorities' | 'mlp';
      mlp?: {
        hiddenSize?: number;
        lr?: number;
        beta1?: number;
        beta2?: number;
        eps?: number;
      };
      jitter?: number;
      mode?: 'parity' | 'relaxed';
      scoreMode?: 'auto' | 'bundle' | 'graph' | 'lb';
      schedulerPolicies?: Array<'height' | 'slack' | 'mix'>;
      schedulerRestarts?: number;
      schedulerSeed?: number;
      schedulerJitter?: number;
      capsSource?: 'slot_limits' | 'spec';
      engineOrder?: string[];
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
  initMode?: 'normal' | 'uniform' | 'zeros' | 'baseline';
  targetMode?: 'normal' | 'uniform' | 'zeros';
  initScale?: number;
  targetScale?: number;
  readbackEvery?: number;
}

export interface EnergyResult {
  backend?: string;
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
  stepsPerRestart?: number;
  bestStep?: number;
  restarts?: number;
  baseline?: {
    cycles: number;
    utilization: number;
    violations: number;
    scheduled: number;
    energy: number;
  };
  schedule?: {
    slotAssignments: Int32Array;
    slotEngines: string[];
    slotIndices: number[];
    duplicates: number;
    missing: number;
  };
  candidates?: Array<{
    restart: number;
    cycles: number;
    utilization: number;
    violations: number;
    steps: number;
  }>;
  taskMeta?: Array<{
    id: number;
    engine: string;
    bundle?: number | null;
    deps: number;
    reads: number;
    writes: number;
  }>;
  scheduler?: string;
  schedulerPolicy?: string;
  schedulerPolicies?: string[];
  scoreMode?: string;
  engineOrder?: string[];
  capsSource?: string;
  mode?: string;
  problem?: EnergyProblem;
}

export interface EnergyStats {
  backend?: string;
  totalTimeMs?: number;
  steps?: number;
  stepTimesMs?: number[];
  energyHistory?: number[];
  readbackCount?: number;
  energy?: number | null;
  energyComponents?: EnergyComponents | null;
  stateStats?: ArrayStats;
}
