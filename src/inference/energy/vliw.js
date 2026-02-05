const ENGINE_ORDER = ['alu', 'valu', 'load', 'store', 'flow'];
const WEIGHT_KEYS = {
  height: 0,
  slack: 1,
  pressure: 2,
  age: 3,
  baseline: 4,
};
const DEFAULT_WEIGHTS = new Float32Array([1.0, 0.6, 0.4, 0.1, 0.2]);

function createRng(seed) {
  let state = seed >>> 0;
  if (!state) state = 0x6d2b79f5;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleNormal(rng) {
  const u = Math.max(rng(), 1e-6);
  const v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function buildTaskIndex(tasks) {
  const byId = new Array(tasks.length);
  tasks.forEach((task) => {
    byId[task.id] = task;
  });
  return byId;
}

function buildHazardDeps(tasks) {
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
      if (prior != null) deps[i].add(prior);
    });
    writes.forEach((addr) => {
      const priorWrite = lastWrite.get(addr);
      if (priorWrite != null) deps[i].add(priorWrite);
      const priorRead = lastRead.get(addr);
      if (priorRead != null) deps[i].add(priorRead);
    });
    reads.forEach((addr) => lastRead.set(addr, i));
    writes.forEach((addr) => {
      lastWrite.set(addr, i);
      lastRead.delete(addr);
    });
  }
  return deps.map((set) => Array.from(set));
}

function buildUnifiedDeps(tasks) {
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

function buildGraph(tasks) {
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

function computeTopologicalOrder(graph) {
  const indeg = graph.indeg.slice();
  const order = [];
  const queue = [];
  for (let i = 0; i < indeg.length; i++) {
    if (indeg[i] === 0) queue.push(i);
  }
  while (queue.length) {
    const node = queue.shift();
    order.push(node);
    const next = graph.succ[node];
    for (let i = 0; i < next.length; i++) {
      const succ = next[i];
      indeg[succ] -= 1;
      if (indeg[succ] === 0) queue.push(succ);
    }
  }
  return order;
}

function computeGraphMetrics(graph) {
  const n = graph.indeg.length;
  const order = computeTopologicalOrder(graph);
  const height = new Float32Array(n);
  const earliest = new Int32Array(n);
  const latest = new Int32Array(n);

  if (order.length !== n) {
    return {
      height,
      slack: new Float32Array(n),
      order,
    };
  }

  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i];
    const next = graph.succ[node];
    let maxChild = 0;
    for (let j = 0; j < next.length; j++) {
      const succ = next[j];
      if (height[succ] > maxChild) maxChild = height[succ];
    }
    height[node] = maxChild + 1;
  }

  for (let i = 0; i < order.length; i++) {
    const node = order[i];
    const next = graph.succ[node];
    for (let j = 0; j < next.length; j++) {
      const succ = next[j];
      const candidate = earliest[node] + 1;
      if (candidate > earliest[succ]) earliest[succ] = candidate;
    }
  }

  let maxPath = 0;
  for (let i = 0; i < n; i++) {
    if (earliest[i] > maxPath) maxPath = earliest[i];
  }
  maxPath += 1;

  latest.fill(maxPath - 1);
  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i];
    const next = graph.succ[node];
    if (!next.length) continue;
    let minLatest = Number.POSITIVE_INFINITY;
    for (let j = 0; j < next.length; j++) {
      const succ = next[j];
      const candidate = latest[succ] - 1;
      if (candidate < minLatest) minLatest = candidate;
    }
    if (Number.isFinite(minLatest)) {
      latest[node] = Math.min(latest[node], minLatest);
    }
  }

  const slack = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const value = latest[i] - earliest[i];
    slack[i] = value >= 0 ? value : 0;
  }

  return {
    height,
    slack,
    order,
  };
}

function buildBaselinePriorities(tasks) {
  const priorities = new Float32Array(tasks.length);
  const bundleCounts = new Map();
  let maxBundle = 0;
  tasks.forEach((task) => {
    const bundle = Number.isFinite(task.bundle) ? task.bundle : 0;
    if (bundle > maxBundle) maxBundle = bundle;
    bundleCounts.set(bundle, (bundleCounts.get(bundle) || 0) + 1);
  });
  let maxCount = 1;
  bundleCounts.forEach((count) => {
    if (count > maxCount) maxCount = count;
  });
  const scale = maxCount + 1;
  const bundleOffsets = new Map();
  tasks.forEach((task) => {
    const bundle = Number.isFinite(task.bundle) ? task.bundle : 0;
    const pos = bundleOffsets.get(bundle) || 0;
    bundleOffsets.set(bundle, pos + 1);
    const rank = bundle * scale + pos;
    priorities[task.id] = -rank;
  });
  return priorities;
}

function computeEngineOffsets(caps) {
  const offsets = {};
  const slotEngines = [];
  const slotIndices = [];
  let cursor = 0;
  ENGINE_ORDER.forEach((engine) => {
    offsets[engine] = cursor;
    const cap = Math.max(0, caps[engine] || 0);
    for (let i = 0; i < cap; i++) {
      slotEngines.push(engine);
      slotIndices.push(i);
    }
    cursor += cap;
  });
  return {
    offsets,
    totalSlots: cursor,
    slotEngines,
    slotIndices,
  };
}

function scheduleWithPriority(tasks, caps, priorities, graph) {
  const n = tasks.length;
  const { succ, indeg } = graph;
  const remaining = indeg.slice();
  const ready = [];
  for (let i = 0; i < n; i++) {
    if (remaining[i] === 0) ready.push(i);
  }

  const {
    offsets,
    totalSlots,
    slotEngines,
    slotIndices,
  } = computeEngineOffsets(caps);
  const gridRows = [];
  const assignmentRows = [];
  let scheduled = 0;
  let usedSlots = 0;
  let cycles = 0;
  let violations = 0;

  while (scheduled < n) {
    if (!ready.length) {
      violations += 1;
      break;
    }
    ready.sort((a, b) => (priorities[b] || 0) - (priorities[a] || 0));
    const slots = { ...caps };
    const usage = {};
    ENGINE_ORDER.forEach((engine) => {
      usage[engine] = 0;
    });
    const row = new Float32Array(totalSlots);
    const assignmentRow = new Int32Array(totalSlots);
    assignmentRow.fill(-1);
    const nextReady = [];
    const scheduledThis = [];

    for (let i = 0; i < ready.length; i++) {
      const taskId = ready[i];
      const task = tasks[taskId];
      const engine = task.engine;
      if (!engine || slots[engine] == null) {
        continue;
      }
      if (slots[engine] > 0) {
        slots[engine] -= 1;
        const slotIndex = offsets[engine] + usage[engine];
        usage[engine] += 1;
        row[slotIndex] = 1;
        assignmentRow[slotIndex] = taskId;
        scheduledThis.push(taskId);
        scheduled += 1;
        usedSlots += 1;
      } else {
        nextReady.push(taskId);
      }
    }

    for (let i = 0; i < scheduledThis.length; i++) {
      const tid = scheduledThis[i];
      const next = succ[tid];
      for (let j = 0; j < next.length; j++) {
        const nid = next[j];
        remaining[nid] -= 1;
        if (remaining[nid] === 0) {
          nextReady.push(nid);
        }
      }
    }

    ready.length = 0;
    ready.push(...nextReady);
    gridRows.push(row);
    assignmentRows.push(assignmentRow);
    cycles += 1;
  }

  const utilization = cycles > 0 && totalSlots > 0
    ? usedSlots / (cycles * totalSlots)
    : 0;

  const grid = new Float32Array(gridRows.length * totalSlots);
  gridRows.forEach((row, rowIndex) => {
    grid.set(row, rowIndex * totalSlots);
  });
  const slotAssignments = new Int32Array(assignmentRows.length * totalSlots);
  assignmentRows.forEach((row, rowIndex) => {
    slotAssignments.set(row, rowIndex * totalSlots);
  });

  return {
    cycles,
    utilization,
    violations,
    scheduled,
    grid,
    gridShape: [gridRows.length, totalSlots, 1],
    slotAssignments,
    slotEngines,
    slotIndices,
  };
}

function scheduleWithHeuristic({
  tasks,
  caps,
  graph,
  features,
  weights,
  basePriorities,
  rng,
  jitter,
}) {
  const n = tasks.length;
  const { succ, indeg } = graph;
  const remaining = indeg.slice();
  const ready = [];
  const age = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    if (remaining[i] === 0) ready.push(i);
  }

  const {
    offsets,
    totalSlots,
    slotEngines,
    slotIndices,
  } = computeEngineOffsets(caps);
  const gridRows = [];
  const assignmentRows = [];
  let scheduled = 0;
  let usedSlots = 0;
  let cycles = 0;
  let violations = 0;

  const scoreById = new Float32Array(n);
  const scheduledFlags = new Uint8Array(n);
  const readyCounts = {};
  ENGINE_ORDER.forEach((engine) => {
    readyCounts[engine] = 0;
  });

  const safeJitter = Number.isFinite(jitter) ? jitter : 0;

  while (scheduled < n) {
    if (!ready.length) {
      violations += 1;
      break;
    }
    ENGINE_ORDER.forEach((engine) => {
      readyCounts[engine] = 0;
    });
    for (let i = 0; i < ready.length; i++) {
      const engine = tasks[ready[i]]?.engine;
      if (engine && readyCounts[engine] != null) {
        readyCounts[engine] += 1;
      }
    }

    const enginePressure = {};
    ENGINE_ORDER.forEach((engine) => {
      const cap = Math.max(0, caps[engine] || 0);
      enginePressure[engine] = cap > 0 ? readyCounts[engine] / cap : 0;
    });

    for (let i = 0; i < ready.length; i++) {
      const taskId = ready[i];
      const task = tasks[taskId];
      const engine = task.engine;
      const pressure = engine ? enginePressure[engine] || 0 : 0;
      const height = features.height[taskId] || 0;
      const slack = features.slack[taskId] || 0;
      const slackScore = -slack;
      const ageScore = age[taskId] || 0;
      const baselineScore = basePriorities ? basePriorities[taskId] || 0 : 0;
      const score = (
        weights[WEIGHT_KEYS.height] * height
        + weights[WEIGHT_KEYS.slack] * slackScore
        + weights[WEIGHT_KEYS.pressure] * pressure
        + weights[WEIGHT_KEYS.age] * ageScore
        + weights[WEIGHT_KEYS.baseline] * baselineScore
      );
      const jitterValue = safeJitter > 0 ? (rng() - 0.5) * safeJitter : 0;
      scoreById[taskId] = score + jitterValue;
    }

    const readyByEngine = {};
    ENGINE_ORDER.forEach((engine) => {
      readyByEngine[engine] = [];
    });
    for (let i = 0; i < ready.length; i++) {
      const taskId = ready[i];
      const engine = tasks[taskId]?.engine;
      if (!engine || readyByEngine[engine] == null) continue;
      readyByEngine[engine].push(taskId);
    }
    ENGINE_ORDER.forEach((engine) => {
      const list = readyByEngine[engine];
      if (list.length > 1) {
        list.sort((a, b) => scoreById[b] - scoreById[a]);
      }
    });

    const engines = ENGINE_ORDER.slice();
    engines.sort((a, b) => enginePressure[b] - enginePressure[a]);

    const slots = { ...caps };
    const usage = {};
    ENGINE_ORDER.forEach((engine) => {
      usage[engine] = 0;
    });
    const row = new Float32Array(totalSlots);
    const assignmentRow = new Int32Array(totalSlots);
    assignmentRow.fill(-1);
    const scheduledThis = [];

    for (let e = 0; e < engines.length; e++) {
      const engine = engines[e];
      let remainingSlots = slots[engine] || 0;
      if (remainingSlots <= 0) continue;
      const list = readyByEngine[engine];
      for (let i = 0; i < list.length && remainingSlots > 0; i++) {
        const taskId = list[i];
        if (remaining[taskId] !== 0) continue;
        remainingSlots -= 1;
        slots[engine] -= 1;
        const slotIndex = offsets[engine] + usage[engine];
        usage[engine] += 1;
        row[slotIndex] = 1;
        assignmentRow[slotIndex] = taskId;
        scheduledThis.push(taskId);
        scheduledFlags[taskId] = 1;
        scheduled += 1;
        usedSlots += 1;
      }
    }

    if (!scheduledThis.length) {
      violations += 1;
      break;
    }

    const nextReady = [];
    for (let i = 0; i < ready.length; i++) {
      const taskId = ready[i];
      if (!scheduledFlags[taskId]) {
        age[taskId] += 1;
        nextReady.push(taskId);
      }
    }
    for (let i = 0; i < scheduledThis.length; i++) {
      const tid = scheduledThis[i];
      scheduledFlags[tid] = 0;
      age[tid] = 0;
      const next = succ[tid];
      for (let j = 0; j < next.length; j++) {
        const nid = next[j];
        remaining[nid] -= 1;
        if (remaining[nid] === 0) {
          age[nid] = 0;
          nextReady.push(nid);
        }
      }
    }

    ready.length = 0;
    ready.push(...nextReady);
    gridRows.push(row);
    assignmentRows.push(assignmentRow);
    cycles += 1;
  }

  const utilization = cycles > 0 && totalSlots > 0
    ? usedSlots / (cycles * totalSlots)
    : 0;

  const grid = new Float32Array(gridRows.length * totalSlots);
  gridRows.forEach((row, rowIndex) => {
    grid.set(row, rowIndex * totalSlots);
  });
  const slotAssignments = new Int32Array(assignmentRows.length * totalSlots);
  assignmentRows.forEach((row, rowIndex) => {
    slotAssignments.set(row, rowIndex * totalSlots);
  });

  return {
    cycles,
    utilization,
    violations,
    scheduled,
    grid,
    gridShape: [gridRows.length, totalSlots, 1],
    slotAssignments,
    slotEngines,
    slotIndices,
  };
}

function initPriorities(count, mode, seed, scale) {
  const priorities = new Float32Array(count);
  const rng = createRng(seed);
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  if (mode === 'baseline') {
    return priorities;
  }
  if (mode === 'zeros') {
    return priorities;
  }
  for (let i = 0; i < count; i++) {
    if (mode === 'uniform') {
      priorities[i] = (rng() * 2 - 1) * safeScale;
    } else {
      priorities[i] = sampleNormal(rng) * safeScale;
    }
  }
  return priorities;
}

function perturbPriorities(base, rng, count, scale) {
  const next = new Float32Array(base);
  const n = next.length;
  const steps = Math.max(1, count);
  for (let i = 0; i < steps; i++) {
    const idx = Math.floor(rng() * n);
    next[idx] += sampleNormal(rng) * scale;
  }
  return next;
}

function initWeights(mode, seed, scale, defaults) {
  const rng = createRng(seed);
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  const weights = new Float32Array(defaults.length);
  if (mode === 'zeros') {
    return weights;
  }
  for (let i = 0; i < defaults.length; i++) {
    weights[i] = defaults[i];
    if (mode === 'baseline') continue;
    if (mode === 'uniform') {
      weights[i] += (rng() * 2 - 1) * safeScale;
    } else {
      weights[i] += sampleNormal(rng) * safeScale;
    }
  }
  return weights;
}

function perturbWeights(base, rng, count, scale) {
  const next = new Float32Array(base);
  const steps = Math.max(1, count);
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  for (let i = 0; i < steps; i++) {
    const idx = Math.floor(rng() * next.length);
    next[idx] += sampleNormal(rng) * safeScale;
  }
  return next;
}

function resolveScheduleEnergy(schedule, taskCount) {
  if (!schedule) return Number.POSITIVE_INFINITY;
  if (schedule.violations > 0) return Number.POSITIVE_INFINITY;
  if (schedule.scheduled < taskCount) return Number.POSITIVE_INFINITY;
  return schedule.cycles;
}

export function runVliwEnergyLoop({
  tasks,
  caps,
  loop,
  search,
  seed,
  initMode,
  initScale,
  diagnostics,
  onProgress,
  onTrace,
}) {
  if (!Array.isArray(tasks) || !tasks.length) {
    throw new Error('VLIW demo requires a non-empty task list.');
  }
  const taskList = buildTaskIndex(tasks);
  const graph = buildGraph(taskList);
  const taskMeta = taskList.map((task) => ({
    id: task.id,
    engine: task.engine,
    bundle: task.bundle ?? null,
    deps: Array.isArray(task.deps) ? task.deps.length : 0,
    reads: Array.isArray(task.reads) ? task.reads.length : 0,
    writes: Array.isArray(task.writes) ? task.writes.length : 0,
  }));
  const maxSteps = Math.max(1, Math.floor(loop?.maxSteps ?? 200));
  const minSteps = Math.max(1, Math.floor(loop?.minSteps ?? 1));
  const stepSize = Number.isFinite(loop?.stepSize) ? loop.stepSize : 0.25;
  const gradientScale = Number.isFinite(loop?.gradientScale) ? loop.gradientScale : 1.0;
  const convergenceThreshold = Number.isFinite(loop?.convergenceThreshold)
    ? loop.convergenceThreshold
    : null;

  const readbackEvery = Math.max(1, Math.floor(diagnostics?.readbackEvery ?? 5));
  const historyLimit = Math.max(1, Math.floor(diagnostics?.historyLimit ?? 200));

  const restarts = Math.max(1, Math.floor(search?.restarts ?? 1));
  let tempStart = Number.isFinite(search?.temperatureStart) ? search.temperatureStart : 2.5;
  const tempDecay = Number.isFinite(search?.temperatureDecay) ? search.temperatureDecay : 0.985;
  const mutationCount = Math.max(1, Math.floor(search?.mutationCount ?? Math.max(1, gradientScale * 4)));
  const policy = search?.policy === 'priorities' ? 'priorities' : 'weights';
  const jitter = Number.isFinite(search?.jitter) ? search.jitter : 0;

  const rng = createRng(seed ?? 1337);
  const baselinePriorities = buildBaselinePriorities(taskList);
  const baselineSchedule = scheduleWithPriority(taskList, caps, baselinePriorities, graph);
  const baselineEnergy = resolveScheduleEnergy(baselineSchedule, taskList.length);
  let bestEnergy = Number.isFinite(baselineEnergy) ? baselineEnergy : Number.POSITIVE_INFINITY;
  let bestSchedule = Number.isFinite(baselineEnergy) ? baselineSchedule : null;
  let bestHistory = [];
  let bestState = bestSchedule ? bestSchedule.grid : null;
  let bestShape = bestSchedule ? bestSchedule.gridShape : null;
  let bestSteps = 0;
  let totalSteps = 0;
  const candidates = [];
  const graphMetrics = computeGraphMetrics(graph);

  const totalStart = performance.now();

  for (let restart = 0; restart < restarts; restart++) {
    const seedValue = Math.floor(rng() * 1e9);
    const priorities = initMode === 'baseline'
      ? new Float32Array(baselinePriorities)
      : initPriorities(taskList.length, initMode, seedValue, initScale);
    const weights = initWeights(initMode, seedValue, initScale, DEFAULT_WEIGHTS);
    let current = policy === 'priorities' ? priorities : weights;
    let currentSchedule = policy === 'priorities'
      ? scheduleWithPriority(taskList, caps, current, graph)
      : scheduleWithHeuristic({
        tasks: taskList,
        caps,
        graph,
        features: graphMetrics,
        weights: current,
        basePriorities: baselinePriorities,
        rng,
        jitter,
      });
    let currentEnergy = resolveScheduleEnergy(currentSchedule, taskList.length);
    let temperature = tempStart;
    const energyHistory = [];
    let restartBestEnergy = currentEnergy;
    let restartBestSchedule = currentSchedule;
    let restartBestSteps = 1;
    let stepsRun = 0;

    if (onTrace) {
      onTrace(0, currentEnergy, {
        cycles: currentSchedule.cycles,
        utilization: currentSchedule.utilization,
      });
    }

    for (let step = 0; step < maxSteps; step++) {
      stepsRun = step + 1;
      const candidate = policy === 'priorities'
        ? perturbPriorities(current, rng, mutationCount, stepSize)
        : perturbWeights(current, rng, mutationCount, stepSize);
      const candidateSchedule = policy === 'priorities'
        ? scheduleWithPriority(taskList, caps, candidate, graph)
        : scheduleWithHeuristic({
          tasks: taskList,
          caps,
          graph,
          features: graphMetrics,
          weights: candidate,
          basePriorities: baselinePriorities,
          rng,
          jitter,
        });
      const candidateEnergy = resolveScheduleEnergy(candidateSchedule, taskList.length);
      const delta = candidateEnergy - currentEnergy;
      const accept = (!Number.isFinite(delta) && candidateEnergy < currentEnergy)
        || delta <= 0
        || rng() < Math.exp(-delta / Math.max(temperature, 1e-6));
      if (accept) {
        current = candidate;
        currentSchedule = candidateSchedule;
        currentEnergy = candidateEnergy;
      }
      if (currentEnergy < restartBestEnergy) {
        restartBestEnergy = currentEnergy;
        restartBestSchedule = currentSchedule;
        restartBestSteps = step + 1;
      }
      if (currentEnergy < bestEnergy) {
        bestEnergy = currentEnergy;
        bestSchedule = currentSchedule;
        bestState = currentSchedule.grid;
        bestShape = currentSchedule.gridShape;
        bestSteps = step + 1;
        if (energyHistory.length) {
          bestHistory = energyHistory.slice();
        }
      }

      if (step % readbackEvery === 0 || step === maxSteps - 1) {
        energyHistory.push(currentEnergy);
        if (energyHistory.length > historyLimit) {
          energyHistory.shift();
        }
      }
      if (onProgress) {
        onProgress({
          stage: 'energy',
          percent: (step + 1) / maxSteps,
          message: `VLIW search ${restart + 1}/${restarts} • step ${step + 1}/${maxSteps}`,
        });
      }
      if (convergenceThreshold != null && step >= minSteps && currentEnergy <= convergenceThreshold) {
        break;
      }
      temperature *= tempDecay;
    }
    totalSteps += stepsRun;

    candidates.push({
      restart: restart + 1,
      cycles: restartBestEnergy,
      utilization: restartBestSchedule.utilization,
      violations: restartBestSchedule.violations,
      steps: restartBestSteps,
    });
  }

  const totalTimeMs = performance.now() - totalStart;
  if (!bestSchedule || !bestState || !bestShape) {
    throw new Error('VLIW search failed to produce a schedule.');
  }

  return {
    steps: totalSteps,
    stepsPerRestart: maxSteps,
    bestStep: bestSteps,
    restarts,
    energy: bestEnergy,
    energyHistory: bestHistory,
    state: bestState,
    shape: bestShape,
    metrics: {
      cycles: bestSchedule.cycles,
      utilization: bestSchedule.utilization,
      violations: bestSchedule.violations,
    },
    baseline: {
      cycles: baselineSchedule.cycles,
      utilization: baselineSchedule.utilization,
      violations: baselineSchedule.violations,
      scheduled: baselineSchedule.scheduled,
      energy: baselineEnergy,
    },
    schedule: {
      slotAssignments: bestSchedule.slotAssignments,
      slotEngines: bestSchedule.slotEngines,
      slotIndices: bestSchedule.slotIndices,
    },
    candidates,
    taskMeta,
    totalTimeMs,
  };
}
