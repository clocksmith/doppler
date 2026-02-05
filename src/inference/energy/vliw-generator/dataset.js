import { DEFAULT_DEPENDENCY_MODEL } from './constants.js';
import { normalizeSpec, resolveCaps } from './spec.js';
import { buildFinalOps } from './finalize.js';
import { buildDeps } from './deps.js';
import { scheduleOpsDep, countCycles } from './schedule.js';

export function buildVliwDatasetFromSpec(specInput, options = {}) {
  const mode = options?.mode === 'parity' ? 'parity' : 'relaxed';
  const spec = normalizeSpec(specInput, { mode });
  const capsMode = options?.capsMode === 'slot_limits' ? 'slot_limits' : 'spec';
  const caps = resolveCaps(spec, { capsMode });
  const useTempDeps = spec.use_temp_deps !== false;
  const dependencyModel = {
    ...DEFAULT_DEPENDENCY_MODEL,
    temp_hazard_tags: useTempDeps,
  };
  const { ops, offloadableCount } = buildFinalOps(spec);
  const { deps, readsList, writesList } = buildDeps(ops, { includeTemp: useTempDeps });
  const tasks = ops.map((op, idx) => ({
    id: idx,
    engine: op.engine,
    reads: readsList[idx],
    writes: writesList[idx],
    deps: deps[idx],
    bundle: null,
  }));
  tasks.forEach((task, idx) => {
    const tempMeta = ops[idx].meta?.temp;
    if (tempMeta) {
      task.temp = Array.isArray(tempMeta) ? tempMeta.slice() : tempMeta;
    }
  });

  const baselineInstrs = scheduleOpsDep(ops, caps, {
    returnOps: true,
    seed: spec.sched_seed ?? 0,
    jitter: spec.sched_jitter ?? 0,
    restarts: spec.sched_restarts ?? 1,
    useTempDeps,
  });
  baselineInstrs.forEach((bundle, cycle) => {
    Object.values(bundle).forEach((slots) => {
      slots.forEach((op) => {
        const task = tasks[op.id];
        if (task) task.bundle = cycle;
      });
    });
  });

  const bundleCount = baselineInstrs.length;
  const baselineCycles = countCycles(baselineInstrs);

  return {
    version: 1,
    label: 'VLIW SIMD schedule (generated)',
    source: 'generated-in-browser',
    spec,
    tasks,
    taskCount: tasks.length,
    bundleCount,
    baselineCycles,
    caps,
    dag: {
      taskCount: tasks.length,
      caps,
      hash: null,
    },
    dependencyModel,
    offloadableCount,
  };
}

export function getDefaultSpec() {
  return normalizeSpec({});
}
