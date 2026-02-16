export function buildDeps(ops: Array<{ meta?: unknown }>): {
  deps: number[][];
  depsLatency: Array<Array<[number, number]>>;
  readsList: number[][];
  writesList: number[][];
};
