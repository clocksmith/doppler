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
}

export interface VliwDependencyModel {
  includes_raw: boolean;
  includes_waw: boolean;
  includes_war: boolean;
  temp_hazard_tags: boolean;
  read_after_read: boolean;
  latency: { default: number };
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

export function buildVliwDatasetFromSpec(specInput: VliwSpec): VliwDataset;
export function getDefaultSpec(): VliwSpec;
