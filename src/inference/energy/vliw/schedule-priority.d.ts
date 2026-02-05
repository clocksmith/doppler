import type { VliwGraph } from './graph.js';
export function scheduleWithPriority(
  tasks: Array<{ id: number; engine: string }>,
  caps: Record<string, number>,
  priorities: Float32Array,
  graph: VliwGraph,
): {
  cycles: number;
  utilization: number;
  violations: number;
  scheduled: number;
  grid: Float32Array;
  gridShape: [number, number, number];
  slotAssignments: Int32Array;
  slotEngines: string[];
  slotIndices: number[];
};

export function scheduleWithHeuristic(params: {
  tasks: Array<{ id: number; engine: string }>;
  caps: Record<string, number>;
  graph: VliwGraph;
  features: { height: Float32Array; slack: Float32Array; order: number[] };
  weights: Float32Array;
  basePriorities?: Float32Array | null;
  rng: () => number;
  jitter?: number;
}): {
  cycles: number;
  utilization: number;
  violations: number;
  scheduled: number;
  grid: Float32Array;
  gridShape: [number, number, number];
  slotAssignments: Int32Array;
  slotEngines: string[];
  slotIndices: number[];
};
