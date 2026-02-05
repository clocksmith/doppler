export interface VliwSpec {
  [key: string]: unknown;
}

export interface VliwTask {
  id: number;
  engine: string;
  reads: number[];
  writes: number[];
  deps: number[];
  bundle: number | null;
  temp?: string | string[];
}

export interface VliwDependencyModel {
  includes_raw: boolean;
  includes_waw: boolean;
  includes_war: boolean;
  temp_hazard_tags: boolean;
  read_after_read: boolean;
  latency: {
    default: number;
    raw?: number;
    waw?: number;
    war?: number;
    temp?: number;
    rar?: number;
  };
}

export interface VliwDataset {
  version: number;
  label: string;
  source: string;
  spec: VliwSpec;
  tasks: VliwTask[];
  taskCount: number;
  bundleCount: number;
  baselineCycles: number;
  caps: Record<string, number>;
  dag: { taskCount: number; caps: Record<string, number>; hash: string | null };
  dependencyModel: VliwDependencyModel;
}

export function buildLayout(specInput: VliwSpec): unknown;
export function buildVliwDatasetFromSpec(
  specInput: VliwSpec,
  options?: { mode?: 'parity' | 'relaxed'; capsMode?: 'slot_limits' | 'spec' },
): VliwDataset;
export function getDefaultSpec(): VliwSpec;
