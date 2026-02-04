const ENGINE_ORDER = ['alu', 'valu', 'load', 'store', 'flow'];

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

function buildGraph(tasks) {
  const n = tasks.length;
  const succ = Array.from({ length: n }, () => []);
  const indeg = new Array(n).fill(0);
  tasks.forEach((task) => {
    if (!Array.isArray(task.deps)) return;
    task.deps.forEach((dep) => {
      if (dep == null) return;
      succ[dep].push(task.id);
      indeg[task.id] += 1;
    });
  });
  return { succ, indeg };
}

function computeEngineOffsets(caps) {
  const offsets = {};
  let cursor = 0;
  ENGINE_ORDER.forEach((engine) => {
    offsets[engine] = cursor;
    cursor += Math.max(0, caps[engine] || 0);
  });
  return { offsets, totalSlots: cursor };
}

function scheduleWithPriority(tasks, caps, priorities, graph) {
  const n = tasks.length;
  const { succ, indeg } = graph;
  const remaining = indeg.slice();
  const ready = [];
  for (let i = 0; i < n; i++) {
    if (remaining[i] === 0) ready.push(i);
  }

  const { offsets, totalSlots } = computeEngineOffsets(caps);
  const gridRows = [];
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
    cycles += 1;
  }

  const utilization = cycles > 0 && totalSlots > 0
    ? usedSlots / (cycles * totalSlots)
    : 0;

  const grid = new Float32Array(gridRows.length * totalSlots);
  gridRows.forEach((row, rowIndex) => {
    grid.set(row, rowIndex * totalSlots);
  });

  return {
    cycles,
    utilization,
    violations,
    grid,
    gridShape: [gridRows.length, totalSlots, 1],
  };
}

function initPriorities(count, mode, seed, scale) {
  const priorities = new Float32Array(count);
  const rng = createRng(seed);
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
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

  const rng = createRng(seed ?? 1337);
  let best = null;
  let bestEnergy = Number.POSITIVE_INFINITY;
  let bestMetrics = null;
  let bestHistory = [];
  let bestState = null;
  let bestShape = null;
  let bestSteps = 0;

  const totalStart = performance.now();

  for (let restart = 0; restart < restarts; restart++) {
    const priorities = initPriorities(taskList.length, initMode, Math.floor(rng() * 1e9), initScale);
    let current = priorities;
    let currentSchedule = scheduleWithPriority(taskList, caps, current, graph);
    let currentEnergy = currentSchedule.cycles;
    let temperature = tempStart;
    const energyHistory = [];

    if (onTrace) {
      onTrace(0, currentEnergy, { cycles: currentEnergy, utilization: currentSchedule.utilization });
    }

    for (let step = 0; step < maxSteps; step++) {
      const candidate = perturbPriorities(current, rng, mutationCount, stepSize);
      const candidateSchedule = scheduleWithPriority(taskList, caps, candidate, graph);
      const candidateEnergy = candidateSchedule.cycles;
      const delta = candidateEnergy - currentEnergy;
      if (delta <= 0 || rng() < Math.exp(-delta / Math.max(temperature, 1e-6))) {
        current = candidate;
        currentSchedule = candidateSchedule;
        currentEnergy = candidateEnergy;
      }
      if (currentEnergy < bestEnergy) {
        bestEnergy = currentEnergy;
        best = current;
        bestMetrics = currentSchedule;
        bestState = currentSchedule.grid;
        bestShape = currentSchedule.gridShape;
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
        bestSteps = step + 1;
        break;
      }
      temperature *= tempDecay;
      bestSteps = step + 1;
    }

    if (best && bestEnergy < Number.POSITIVE_INFINITY && energyHistory.length) {
      bestHistory = energyHistory;
    }
  }

  const totalTimeMs = performance.now() - totalStart;
  if (!bestMetrics || !bestState || !bestShape) {
    throw new Error('VLIW search failed to produce a schedule.');
  }

  return {
    steps: bestSteps,
    energy: bestEnergy,
    energyHistory: bestHistory,
    state: bestState,
    shape: bestShape,
    metrics: {
      cycles: bestMetrics.cycles,
      utilization: bestMetrics.utilization,
      violations: bestMetrics.violations,
    },
    totalTimeMs,
  };
}
