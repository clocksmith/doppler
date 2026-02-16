import { ENGINE_ORDER } from './constants.js';
import { computeEngineOffsets } from './schedule-engine.js';

export function scheduleFromBundles(tasks, caps) {
  const n = tasks.length;
  const {
    offsets,
    totalSlots,
    totalSlotsNonDebug,
    slotEngines,
    slotIndices,
  } = computeEngineOffsets(caps);
  const buckets = new Map();
  let maxBundle = -1;
  let hasBundles = false;

  tasks.forEach((task) => {
    if (!task) return;
    const bundle = Number.isFinite(task.bundle) ? task.bundle : null;
    if (bundle == null || bundle < 0) return;
    hasBundles = true;
    if (bundle > maxBundle) maxBundle = bundle;
    let byEngine = buckets.get(bundle);
    if (!byEngine) {
      byEngine = new Map();
      buckets.set(bundle, byEngine);
    }
    const engine = task.engine;
    if (!engine) return;
    const list = byEngine.get(engine) || [];
    list.push(task.id);
    byEngine.set(engine, list);
  });

  if (!hasBundles) return null;

  const gridRows = [];
  const assignmentRows = [];
  let cyclesUsed = 0;
  let scheduled = 0;
  let duplicates = 0;
  let usedSlots = 0;
  let usedSlotsNonDebug = 0;
  let violations = 0;
  const taskCycles = new Int32Array(n);
  taskCycles.fill(-1);

  for (let cycle = 0; cycle <= maxBundle; cycle++) {
    const row = new Float32Array(totalSlots);
    const assignmentRow = new Int32Array(totalSlots);
    assignmentRow.fill(-1);
    const byEngine = buckets.get(cycle) || new Map();
    let rowUsed = false;
    ENGINE_ORDER.forEach((engine) => {
      const list = byEngine.get(engine) || [];
      const cap = Math.max(0, caps[engine] || 0);
      for (let i = 0; i < list.length; i++) {
        if (i >= cap) {
          violations += 1;
          continue;
        }
        const taskId = list[i];
        const slotIndex = offsets[engine] + i;
        row[slotIndex] = 1;
        assignmentRow[slotIndex] = taskId;
        rowUsed = true;
        if (taskCycles[taskId] !== -1) {
          duplicates += 1;
        } else {
          taskCycles[taskId] = cycle;
        }
        scheduled += 1;
        usedSlots += 1;
        if (engine !== 'debug') {
          usedSlotsNonDebug += 1;
        }
      }
    });
    if (rowUsed) cyclesUsed += 1;
    gridRows.push(row);
    assignmentRows.push(assignmentRow);
  }

  let missing = 0;
  for (let i = 0; i < taskCycles.length; i++) {
    if (taskCycles[i] === -1) missing += 1;
  }
  const cycles = cyclesUsed;
  const utilization = cyclesUsed > 0 && totalSlotsNonDebug > 0
    ? usedSlotsNonDebug / (cyclesUsed * totalSlotsNonDebug)
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
    duplicates,
    missing,
    grid,
    gridShape: [gridRows.length, totalSlots, 1],
    slotAssignments,
    slotEngines,
    slotIndices,
  };
}
