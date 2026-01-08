import type {
  VliwEnergyDiagnostics,
  VliwEnergyLoopConfig,
  VliwEnergyResult,
  VliwSearchConfig,
  VliwTask,
} from '../vliw.js';

export declare function runVliwEnergyLoop(input: {
  tasks: VliwTask[];
  caps: Record<string, number>;
  dependencyModel?: {
    includes_raw?: boolean;
    includes_waw?: boolean;
    includes_war?: boolean;
    temp_hazard_tags?: boolean;
    read_after_read?: boolean;
    latency?: {
      default?: number;
      raw?: number;
      waw?: number;
      war?: number;
      temp?: number;
      rar?: number;
    };
  };
  loop?: VliwEnergyLoopConfig;
  search?: VliwSearchConfig;
  seed?: number;
  initMode?: 'normal' | 'uniform' | 'zeros' | 'baseline';
  initScale?: number;
  diagnostics?: VliwEnergyDiagnostics;
  onProgress?: (payload: { stage?: string; percent: number; message?: string }) => void;
  onTrace?: (step: number, energy: number, metrics: Record<string, number>) => void;
}): Promise<VliwEnergyResult>;
