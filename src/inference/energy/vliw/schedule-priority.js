import { ENGINE_ORDER, WEIGHT_KEYS } from './constants.js';
import { computeEngineOffsets } from './schedule-engine.js';

export function scheduleWithPriority(tasks, caps, priorities, graph) {
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
  let duplicates = 0;
  let usedSlots = 0;
  let cycles = 0;
  let violations = 0;
  const taskCycles = new Int32Array(n);
  taskCycles.fill(-1);

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
        if (taskCycles[taskId] !== -1) {
          duplicates += 1;
        } else {
          taskCycles[taskId] = cycles;
        }
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
  let missing = 0;
  for (let i = 0; i < taskCycles.length; i++) {
    if (taskCycles[i] === -1) missing += 1;
  }

  return {
    cycles,
    utilization,
    violations,
    scheduled,
    duplicates,
    missing,
    grid,
    gridShape: [gridRows.length, totalSlots, 1],
    slotAssignments,
    slotEngines,
    slotIndices,
  };
}

export function scheduleWithHeuristic({
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
  let duplicates = 0;
  let usedSlots = 0;
  let cycles = 0;
  let violations = 0;
  const taskCycles = new Int32Array(n);
  taskCycles.fill(-1);

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
        if (taskCycles[taskId] !== -1) {
          duplicates += 1;
        } else {
          taskCycles[taskId] = cycles;
        }
        scheduled += 1;
        usedSlots += 1;
        scheduledFlags[taskId] = 1;
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
  let missing = 0;
  for (let i = 0; i < taskCycles.length; i++) {
    if (taskCycles[i] === -1) missing += 1;
  }

  return {
    cycles,
    utilization,
    violations,
    scheduled,
    duplicates,
    missing,
    grid,
    gridShape: [gridRows.length, totalSlots, 1],
    slotAssignments,
    slotEngines,
    slotIndices,
  };
}
