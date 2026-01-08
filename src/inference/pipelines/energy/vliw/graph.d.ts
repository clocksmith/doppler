export interface VliwGraph {
  succ: number[][];
  indeg: number[];
  deps: number[][];
}
export function buildGraph(tasks: Array<{ id: number; deps?: number[]; reads?: number[]; writes?: number[] }>): VliwGraph;
