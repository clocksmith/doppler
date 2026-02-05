import { createRng } from './rng.js';
import { computeEngineOffsets } from './schedule-engine.js';
import { ENGINE_ORDER as DEFAULT_ENGINE_ORDER } from './constants.js';

export function buildLatencyDeps(tasks, dependencyModel = null) {
  const n = tasks.length;
  const depsLatency = Array.from({ length: n }, () => []);
  const lastWrite = new Map();
  const lastRead = new Map();
  const lastTemp = new Map();
  const model = dependencyModel || {};
  const latency = model.latency || {};
  const rawLatency = Number.isFinite(latency.raw) ? latency.raw : (latency.default ?? 1);
  const wawLatency = Number.isFinite(latency.waw) ? latency.waw : (latency.default ?? 1);
  const warLatency = Number.isFinite(latency.war) ? latency.war : (latency.default ?? 1);
  const tempLatency = Number.isFinite(latency.temp) ? latency.temp : (latency.default ?? 1);
  const rarLatency = Number.isFinite(latency.rar) ? latency.rar : (latency.default ?? 1);
  const includeRaw = model.includes_raw !== false;
  const includeWaw = model.includes_waw !== false;
  const includeWar = model.includes_war !== false;
  const includeTemp = model.temp_hazard_tags !== false;
  const includeRar = model.read_after_read === true;

  for (let i = 0; i < n; i++) {
    const task = tasks[i];
    if (!task) continue;
    const reads = Array.isArray(task.reads) ? task.reads : [];
    const writes = Array.isArray(task.writes) ? task.writes : [];
    reads.forEach((addr) => {
      if (includeRaw) {
        const dep = lastWrite.get(addr);
        if (dep != null && dep !== i) depsLatency[i].push([dep, rawLatency]);
      }
      if (includeRar) {
        const depRead = lastRead.get(addr);
        if (depRead != null && depRead !== i) depsLatency[i].push([depRead, rarLatency]);
      }
    });
    writes.forEach((addr) => {
      if (includeWaw) {
        const depWrite = lastWrite.get(addr);
        if (depWrite != null && depWrite !== i) depsLatency[i].push([depWrite, wawLatency]);
      }
      if (includeWar) {
        const depRead = lastRead.get(addr);
        if (depRead != null && depRead !== i) depsLatency[i].push([depRead, warLatency]);
      }
    });
    const tempMeta = task.temp ?? task.meta?.temp;
    let temps = [];
    if (tempMeta) {
      temps = typeof tempMeta === 'string' ? [tempMeta] : Array.from(tempMeta);
    }
    if (includeTemp) {
      temps.forEach((key) => {
        const dep = lastTemp.get(key);
        if (dep != null && dep !== i) depsLatency[i].push([dep, tempLatency]);
      });
    }

    reads.forEach((addr) => lastRead.set(addr, i));
    writes.forEach((addr) => {
      lastWrite.set(addr, i);
      lastRead.delete(addr);
    });
    temps.forEach((key) => lastTemp.set(key, i));
  }

  const merged = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    const seen = new Set();
    depsLatency[i].forEach(([dep, depLatency]) => {
      merged[i].push([dep, depLatency]);
      seen.add(dep);
    });
    const explicit = Array.isArray(tasks[i]?.deps) ? tasks[i].deps : [];
    explicit.forEach((dep) => {
      if (dep == null || dep < 0 || dep >= n) return;
      if (seen.has(dep)) return;
      merged[i].push([dep, Number.isFinite(latency.default) ? latency.default : 1]);
      seen.add(dep);
    });
  }

  return merged;
}

function computeGraphStats(children, indegree) {
  const n = indegree.length;
  const indegWork = indegree.slice();
  const topo = [];
  const queue = [];
  for (let i = 0; i < n; i++) {
    if (indegWork[i] === 0) queue.push(i);
  }
  while (queue.length) {
    const node = queue.pop();
    topo.push(node);
    const next = children[node];
    for (let i = 0; i < next.length; i++) {
      const [succ] = next[i];
      indegWork[succ] -= 1;
      if (indegWork[succ] === 0) queue.push(succ);
    }
  }

  const earliestStatic = new Int32Array(n);
  for (let i = 0; i < topo.length; i++) {
    const node = topo[i];
    const next = children[node];
    for (let j = 0; j < next.length; j++) {
      const [succ, latency] = next[j];
      const candidate = earliestStatic[node] + latency;
      if (candidate > earliestStatic[succ]) earliestStatic[succ] = candidate;
    }
  }

  const height = new Int32Array(n);
  height.fill(1);
  for (let i = topo.length - 1; i >= 0; i--) {
    const node = topo[i];
    const next = children[node];
    if (!next.length) continue;
    let maxChild = 0;
    for (let j = 0; j < next.length; j++) {
      const [succ, latency] = next[j];
      const candidate = height[succ] + latency;
      if (candidate > maxChild) maxChild = candidate;
    }
    height[node] = 1 + maxChild;
  }

  let criticalPath = 0;
  for (let i = 0; i < n; i++) {
    const value = earliestStatic[i] + height[i] - 1;
    if (value > criticalPath) criticalPath = value;
  }

  const slack = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    slack[i] = criticalPath - (earliestStatic[i] + height[i] - 1);
  }

  const outdeg = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    outdeg[i] = children[i].length;
  }

  return {
    earliestStatic,
    height,
    slack,
    outdeg,
  };
}

function buildEmptyRow(totalSlots) {
  const row = new Float32Array(totalSlots);
  const assignmentRow = new Int32Array(totalSlots);
  assignmentRow.fill(-1);
  return { row, assignmentRow };
}

export function scheduleGraphOnce(
  tasks,
  caps,
  {
    seed = 0,
    jitter = 0,
    dependencyModel = null,
    policy = 'mix',
    engineOrder = DEFAULT_ENGINE_ORDER,
  } = {},
) {
  const n = tasks.length;
  if (!n) {
    return {
      cycles: 0,
      utilization: 0,
      violations: 0,
      scheduled: 0,
      duplicates: 0,
      missing: 0,
      grid: new Float32Array(0),
      gridShape: [0, 0, 1],
      slotAssignments: new Int32Array(0),
      slotEngines: [],
      slotIndices: [],
    };
  }

  const depsLatency = buildLatencyDeps(tasks, dependencyModel);
  const children = Array.from({ length: n }, () => []);
  const indegree = Array.from({ length: n }, () => 0);
  for (let i = 0; i < n; i++) {
    depsLatency[i].forEach(([dep, latency]) => {
      if (dep == null || dep < 0 || dep >= n) return;
      children[dep].push([i, latency]);
      indegree[i] += 1;
    });
  }

  const { earliestStatic, height, slack, outdeg } = computeGraphStats(children, indegree);
  const rng = jitter > 0 ? createRng(seed) : null;

  const prioTuple = (tid) => {
    const jitterVal = rng ? rng() * jitter : 0;
    if (policy === 'height') {
      return [-height[tid], jitterVal];
    }
    if (policy === 'slack') {
      return [slack[tid], -height[tid], jitterVal];
    }
    if (policy === 'mix') {
      return [slack[tid], -height[tid], -outdeg[tid], jitterVal];
    }
    return [slack[tid], -height[tid], -outdeg[tid], jitterVal];
  };

  const earliest = Array.from({ length: n }, () => 0);
  const scheduled = Array.from({ length: n }, () => -1);
  const ready = new Set();
  for (let i = 0; i < n; i++) {
    if (indegree[i] === 0) ready.add(i);
  }

  const {
    offsets,
    totalSlots,
    totalSlotsNonDebug,
    slotEngines,
    slotIndices,
  } = computeEngineOffsets(caps);
  const gridRows = [];
  const assignmentRows = [];
  const taskCycles = new Int32Array(n);
  taskCycles.fill(-1);
  let remaining = n;
  let cycle = 0;
  let usedSlots = 0;
  let usedSlotsNonDebug = 0;
  let duplicates = 0;
  let violations = 0;

  while (remaining > 0) {
    const { row, assignmentRow } = buildEmptyRow(totalSlots);
    const writesCycle = new Set();
    const engineCounts = {};
    const scheduledNow = [];

    for (let e = 0; e < engineOrder.length; e++) {
      const engine = engineOrder[e];
      const cap = caps?.[engine] ?? 0;
      if (cap <= 0) continue;
      const candidates = [];
      ready.forEach((tid) => {
        if (scheduled[tid] >= 0) return;
        if (earliest[tid] > cycle) return;
        if (tasks[tid]?.engine !== engine) return;
        candidates.push(tid);
      });
      if (candidates.length) {
        candidates.sort((a, b) => {
          const pa = prioTuple(a);
          const pb = prioTuple(b);
          for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
            if (pa[i] < pb[i]) return -1;
            if (pa[i] > pb[i]) return 1;
          }
          return pa.length - pb.length;
        });
      }
      for (let i = 0; i < candidates.length && (engineCounts[engine] || 0) < cap; i++) {
        const tid = candidates[i];
        const writes = Array.isArray(tasks[tid]?.writes) ? tasks[tid].writes : [];
        if (writes.some((addr) => writesCycle.has(addr))) continue;
        const slotIndex = offsets[engine] + (engineCounts[engine] || 0);
        row[slotIndex] = 1;
        assignmentRow[slotIndex] = tid;
        scheduled[tid] = cycle;
        scheduledNow.push(tid);
        engineCounts[engine] = (engineCounts[engine] || 0) + 1;
        writes.forEach((addr) => writesCycle.add(addr));
        usedSlots += 1;
        if (engine !== 'debug') {
          usedSlotsNonDebug += 1;
        }
      }
    }

    gridRows.push(row);
    assignmentRows.push(assignmentRow);

    if (!scheduledNow.length) {
      if (ready.size === 0) {
        violations += 1;
        break;
      }
      let nextCycle = null;
      ready.forEach((tid) => {
        if (scheduled[tid] >= 0) return;
        const rt = earliest[tid];
        if (nextCycle == null || rt < nextCycle) nextCycle = rt;
      });
      if (nextCycle == null || nextCycle <= cycle) {
        cycle += 1;
        continue;
      }
      for (let c = cycle + 1; c < nextCycle; c++) {
        const empty = buildEmptyRow(totalSlots);
        gridRows.push(empty.row);
        assignmentRows.push(empty.assignmentRow);
      }
      cycle = nextCycle;
      continue;
    }

    scheduledNow.forEach((tid) => {
      remaining -= 1;
      ready.delete(tid);
      if (taskCycles[tid] !== -1) {
        duplicates += 1;
      } else {
        taskCycles[tid] = cycle;
      }
      const next = children[tid];
      for (let i = 0; i < next.length; i++) {
        const [child, latency] = next[i];
        indegree[child] -= 1;
        earliest[child] = Math.max(earliest[child], cycle + latency);
        if (indegree[child] === 0) {
          ready.add(child);
        }
      }
    });

    cycle += 1;
  }

  let missing = 0;
  for (let i = 0; i < taskCycles.length; i++) {
    if (taskCycles[i] === -1) missing += 1;
  }
  const cycles = gridRows.length;
  const utilization = cycles > 0 && totalSlotsNonDebug > 0
    ? usedSlotsNonDebug / (cycles * totalSlotsNonDebug)
    : 0;

  const grid = new Float32Array(gridRows.length * totalSlots);
  gridRows.forEach((row, rowIndex) => {
    grid.set(row, rowIndex * totalSlots);
  });
  const slotAssignments = new Int32Array(assignmentRows.length * totalSlots);
  assignmentRows.forEach((row, rowIndex) => {
    slotAssignments.set(row, rowIndex * totalSlots);
  });

  if (missing > 0) {
    violations += 1;
  }

  return {
    cycles,
    utilization,
    violations,
    scheduled: n - missing,
    duplicates,
    missing,
    grid,
    gridShape: [gridRows.length, totalSlots, 1],
    slotAssignments,
    slotEngines,
    slotIndices,
  };
}

export function scheduleGraphWithRestarts(
  tasks,
  caps,
  {
    seed = 0,
    jitter = 0,
    restarts = 1,
    dependencyModel = null,
    policy = 'mix',
    engineOrder = DEFAULT_ENGINE_ORDER,
  } = {},
) {
  const safeRestarts = Math.max(1, Math.floor(restarts));
  let bestSchedule = null;
  for (let k = 0; k < safeRestarts; k++) {
    const schedule = scheduleGraphOnce(tasks, caps, {
      seed: seed + k,
      jitter,
      dependencyModel,
      policy,
      engineOrder,
    });
    if (!bestSchedule || schedule.cycles < bestSchedule.cycles) {
      bestSchedule = schedule;
    }
  }
  return bestSchedule;
}

export function scheduleGraphWithPolicies(
  tasks,
  caps,
  {
    policies = ['mix'],
    restarts = 1,
    seed = 0,
    jitter = 0,
    dependencyModel = null,
    engineOrder = DEFAULT_ENGINE_ORDER,
  } = {},
) {
  const safePolicies = Array.isArray(policies) && policies.length ? policies : ['mix'];
  let bestSchedule = null;
  let bestPolicy = safePolicies[0];
  safePolicies.forEach((policy) => {
    const schedule = scheduleGraphWithRestarts(tasks, caps, {
      seed,
      jitter,
      restarts,
      dependencyModel,
      policy,
      engineOrder,
    });
    if (!schedule) return;
    if (!bestSchedule || schedule.cycles < bestSchedule.cycles) {
      bestSchedule = schedule;
      bestPolicy = policy;
    }
  });
  return {
    schedule: bestSchedule,
    policy: bestPolicy,
  };
}

export function scheduleWithLatency(tasks, caps, options = {}) {
  return scheduleGraphOnce(tasks, caps, options);
}
