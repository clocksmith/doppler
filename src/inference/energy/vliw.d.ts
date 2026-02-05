/**
 * VLIW Energy Demo Helpers
 *
 * @module inference/energy/vliw
 */

export interface VliwTask {
  id: number;
  engine: string;
  reads?: number[];
  writes?: number[];
  deps?: number[];
  bundle?: number;
}

export interface VliwSearchConfig {
  restarts?: number;
  temperatureStart?: number;
  temperatureDecay?: number;
  mutationCount?: number;
  policy?: 'weights' | 'priorities';
  jitter?: number;
}

export interface VliwEnergyLoopConfig {
  maxSteps?: number;
  minSteps?: number;
  stepSize?: number;
  gradientScale?: number;
  convergenceThreshold?: number | null;
}

export interface VliwEnergyDiagnostics {
  readbackEvery?: number;
  historyLimit?: number;
}

export interface VliwEnergyResult {
  steps: number;
  energy: number;
  energyHistory: number[];
  state: Float32Array;
  shape: number[];
  metrics: {
    cycles: number;
    utilization: number;
    violations: number;
  };
  baseline: {
    cycles: number;
    utilization: number;
    violations: number;
    scheduled: number;
    energy: number;
  };
  stepsPerRestart: number;
  bestStep: number;
  restarts: number;
  schedule: {
    slotAssignments: Int32Array;
    slotEngines: string[];
    slotIndices: number[];
  };
  candidates: Array<{
    restart: number;
    cycles: number;
    utilization: number;
    violations: number;
    steps: number;
  }>;
  taskMeta: Array<{
    id: number;
    engine: string;
    bundle?: number | null;
    deps: number;
    reads: number;
    writes: number;
  }>;
  totalTimeMs: number;
}

export declare function runVliwEnergyLoop(input: {
  tasks: VliwTask[];
  caps: Record<string, number>;
  loop?: VliwEnergyLoopConfig;
  search?: VliwSearchConfig;
  seed?: number;
  initMode?: 'normal' | 'uniform' | 'zeros' | 'baseline';
  initScale?: number;
  diagnostics?: VliwEnergyDiagnostics;
  onProgress?: (payload: { stage?: string; percent: number; message?: string }) => void;
  onTrace?: (step: number, energy: number, metrics: Record<string, number>) => void;
}): VliwEnergyResult;
