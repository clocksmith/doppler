import { buildTaskIndex } from './tasks.js';

export function buildHazardDeps(tasks) {
  const byId = buildTaskIndex(tasks);
  const n = tasks.length;
  const deps = Array.from({ length: n }, () => new Set());
  const lastWrite = new Map();
  const lastRead = new Map();
  for (let i = 0; i < n; i++) {
    const task = byId[i];
    if (!task) continue;
    const reads = Array.isArray(task.reads) ? task.reads : [];
    const writes = Array.isArray(task.writes) ? task.writes : [];
    reads.forEach((addr) => {
      const prior = lastWrite.get(addr);
      if (prior != null && prior !== i) deps[i].add(prior);
    });
    writes.forEach((addr) => {
      const priorWrite = lastWrite.get(addr);
      if (priorWrite != null && priorWrite !== i) deps[i].add(priorWrite);
      const priorRead = lastRead.get(addr);
      if (priorRead != null && priorRead !== i) deps[i].add(priorRead);
    });
    reads.forEach((addr) => lastRead.set(addr, i));
    writes.forEach((addr) => {
      lastWrite.set(addr, i);
      lastRead.delete(addr);
    });
  }
  return deps.map((set) => Array.from(set));
}

export function buildUnifiedDeps(tasks) {
  const n = tasks.length;
  const merged = Array.from({ length: n }, () => new Set());
  tasks.forEach((task) => {
    if (!task || task.id == null) return;
    const deps = Array.isArray(task.deps) ? task.deps : [];
    deps.forEach((dep) => {
      if (dep != null) merged[task.id].add(dep);
    });
  });
  const hazardDeps = buildHazardDeps(tasks);
  hazardDeps.forEach((deps, id) => {
    deps.forEach((dep) => merged[id].add(dep));
  });
  return merged.map((set) => Array.from(set));
}
