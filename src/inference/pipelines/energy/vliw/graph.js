import { buildUnifiedDeps } from './hazards.js';

export function buildGraph(tasks) {
  const n = tasks.length;
  const deps = buildUnifiedDeps(tasks);
  const succ = Array.from({ length: n }, () => []);
  const indeg = new Array(n).fill(0);
  for (let id = 0; id < n; id++) {
    const depsList = deps[id];
    for (let i = 0; i < depsList.length; i++) {
      const dep = depsList[i];
      if (dep == null || dep < 0 || dep >= n) continue;
      succ[dep].push(id);
      indeg[id] += 1;
    }
  }
  return { succ, indeg, deps };
}
