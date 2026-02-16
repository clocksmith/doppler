export function buildLatencyDeps(
  tasks: Array<{ id: number; deps?: number[]; reads?: number[]; writes?: number[]; temp?: string | string[] }>,
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
  },
): Array<Array<[number, number]>>;

export function scheduleGraphOnce(
  tasks: Array<{ id: number; engine: string; reads?: number[]; writes?: number[]; deps?: number[] }>,
  caps: Record<string, number>,
  options?: {
    seed?: number;
    jitter?: number;
    dependencyModel?: Record<string, unknown> | null;
    policy?: 'height' | 'slack' | 'mix';
    engineOrder?: string[];
  },
): {
  cycles: number;
  utilization: number;
  violations: number;
  scheduled: number;
  duplicates: number;
  missing: number;
  grid: Float32Array;
  gridShape: [number, number, number];
  slotAssignments: Int32Array;
  slotEngines: string[];
  slotIndices: number[];
};

export function scheduleGraphWithRestarts(
  tasks: Array<{ id: number; engine: string; reads?: number[]; writes?: number[]; deps?: number[] }>,
  caps: Record<string, number>,
  options?: {
    seed?: number;
    jitter?: number;
    restarts?: number;
    dependencyModel?: Record<string, unknown> | null;
    policy?: 'height' | 'slack' | 'mix';
    engineOrder?: string[];
  },
): ReturnType<typeof scheduleGraphOnce> | null;

export function scheduleGraphWithPolicies(
  tasks: Array<{ id: number; engine: string; reads?: number[]; writes?: number[]; deps?: number[] }>,
  caps: Record<string, number>,
  options?: {
    policies?: Array<'height' | 'slack' | 'mix'>;
    restarts?: number;
    seed?: number;
    jitter?: number;
    dependencyModel?: Record<string, unknown> | null;
    engineOrder?: string[];
  },
): { schedule: ReturnType<typeof scheduleGraphOnce> | null; policy: string };

export function scheduleWithLatency(
  tasks: Array<{ id: number; engine: string; reads?: number[]; writes?: number[]; deps?: number[] }>,
  caps: Record<string, number>,
  options?: {
    seed?: number;
    jitter?: number;
    dependencyModel?: Record<string, unknown> | null;
    policy?: 'height' | 'slack' | 'mix';
    engineOrder?: string[];
  },
): ReturnType<typeof scheduleGraphOnce>;
