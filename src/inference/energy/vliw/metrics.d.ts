import type { VliwGraph } from './graph.js';
export function computeTopologicalOrder(graph: VliwGraph): number[];
export function computeGraphMetrics(graph: VliwGraph): {
  height: Float32Array;
  slack: Float32Array;
  order: number[];
};
