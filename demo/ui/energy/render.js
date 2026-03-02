import { $, setText } from '../dom.js';
import { formatMs, formatScalar } from '../format.js';

function formatSpecValue(value) {
  if (Array.isArray(value)) return `[${value.join(',')}]`;
  if (value && typeof value === 'object') return JSON.stringify(value);
  if (value == null) return 'null';
  return String(value);
}

function formatSpecLines(spec) {
  if (!spec || typeof spec !== 'object') return [];
  const keys = Object.keys(spec).sort();
  return keys.map((key) => `  ${key}: ${formatSpecValue(spec[key])}`);
}

function formatVliwSlotLabel(engine, slotIndex) {
  if (!engine) return `slot${slotIndex}`;
  return `${engine}${slotIndex}`;
}

export function clearEnergyBoard() {
  const board = $('energy-board');
  if (!board) return;
  board.innerHTML = '';
  clearEnergyVector();
  clearEnergyIntensityBoard();
  clearEnergyKernelSummary();
  clearEnergyBundleView();
}

export function clearEnergyVector() {
  const vector = $('energy-vector');
  if (vector) vector.textContent = '';
}

export function clearEnergyIntensityBoard() {
  const board = $('energy-board-intensity');
  if (board) board.innerHTML = '';
}

export function clearEnergyKernelSummary() {
  const summary = $('energy-kernel-summary');
  if (!summary) return;
  summary.textContent = '';
}

export function clearEnergyBundleView() {
  const view = $('energy-bundle-view');
  if (view) view.textContent = '';
  const select = $('energy-vliw-bundle-select');
  if (select) select.innerHTML = '';
}

function resolveEnergyGrid(shapeOrSize) {
  if (Array.isArray(shapeOrSize)) return shapeOrSize;
  if (Number.isFinite(shapeOrSize)) return [shapeOrSize, shapeOrSize];
  return [0, 0];
}

export function renderEnergyBoard(state, shapeOrSize, threshold) {
  const board = $('energy-board');
  if (!board || !state) return;
  const [rows, cols] = resolveEnergyGrid(shapeOrSize);
  if (!rows || !cols) return;
  board.innerHTML = '';
  board.style.setProperty('--energy-grid-size', `${cols}`);
  for (let i = 0; i < rows * cols; i++) {
    const cell = document.createElement('div');
    cell.className = 'energy-cell';
    const value = state[i] || 0;
    const on = threshold != null ? value >= threshold : value > 0.5;
    if (on) cell.classList.add('is-on');
    board.appendChild(cell);
  }
}

export function renderEnergyVector(state, rows, cols, threshold) {
  const vector = $('energy-vector');
  if (!vector || !state) return;
  const values = [];
  for (let row = 0; row < rows; row++) {
    const rowValues = [];
    for (let col = 0; col < cols; col++) {
      const idx = row * cols + col;
      const value = state[idx] || 0;
      rowValues.push(threshold != null ? (value >= threshold ? 1 : 0) : value);
    }
    values.push(rowValues);
  }
  vector.textContent = values.map((row) => row.join(' ')).join('\n');
}

export function renderEnergyIntensityBoard(state, rows, cols) {
  const board = $('energy-board-intensity');
  if (!board || !state) return;
  board.innerHTML = '';
  const gridCols = Math.min(cols, 128);
  board.style.setProperty('--energy-grid-size', `${gridCols}`);
  for (let i = 0; i < rows * cols; i++) {
    const cell = document.createElement('div');
    cell.className = 'energy-cell';
    const value = Math.min(1, Math.max(0, state[i] || 0));
    cell.style.opacity = String(0.15 + value * 0.85);
    board.appendChild(cell);
  }
}

export function renderVliwKernelSummary(summary, datasetMeta) {
  const summaryEl = $('energy-kernel-summary');
  if (!summaryEl) return;
  if (!summary) {
    summaryEl.textContent = '';
    return;
  }
  const lines = [];
  if (datasetMeta?.label) {
    lines.push(`Dataset: ${datasetMeta.label}`);
  }
  if (Number.isFinite(datasetMeta?.bundleCount)) {
    lines.push(`Bundles: ${datasetMeta.bundleCount}`);
  }
  if (Number.isFinite(datasetMeta?.taskCount)) {
    lines.push(`Tasks: ${datasetMeta.taskCount}`);
  }
  if (Number.isFinite(datasetMeta?.baselineCycles)) {
    lines.push(`Baseline cycles: ${datasetMeta.baselineCycles}`);
  }
  if (Number.isFinite(summary.bestCycles)) {
    lines.push(`Best cycles: ${summary.bestCycles}`);
  }
  if (Number.isFinite(summary.utilization)) {
    lines.push(`Utilization: ${formatScalar(summary.utilization, 4)}`);
  }
  if (summary.mode) {
    lines.push(`Mode: ${summary.mode}`);
  }
  if (summary.scoreMode) {
    lines.push(`Score mode: ${summary.scoreMode}`);
  }
  if (summary.scheduler) {
    lines.push(`Scheduler: ${summary.scheduler}`);
  }
  if (summary.schedulerPolicy) {
    lines.push(`Scheduler policy: ${summary.schedulerPolicy}`);
  }
  if (summary.mlpStats) {
    const mlp = summary.mlpStats;
    if (Number.isFinite(mlp.hiddenSize) && Number.isFinite(mlp.lr)) {
      lines.push(`MLP: hidden ${mlp.hiddenSize} • lr ${formatScalar(mlp.lr, 6)} • train ${mlp.trainSteps} • failures ${mlp.trainFailures}`);
    } else {
      lines.push(`MLP: train ${mlp.trainSteps} • failures ${mlp.trainFailures}`);
    }
    if (mlp.firstError) {
      lines.push(`MLP first error: ${mlp.firstError}`);
    }
  }
  if (Array.isArray(summary.schedulerPolicies) && summary.schedulerPolicies.length) {
    lines.push(`Scheduler policies: ${summary.schedulerPolicies.join(', ')}`);
  }
  if (Array.isArray(summary.engineOrder) && summary.engineOrder.length) {
    lines.push(`Engine order: ${summary.engineOrder.join(', ')}`);
  }
  if (summary.capsSource) {
    lines.push(`Caps: ${summary.capsSource}`);
  }
  if (Number.isFinite(summary.bundleLimit)) {
    lines.push(`Bundle limit: ${summary.bundleLimit}`);
  }
  if (summary.baseline) {
    const baselineCycles = summary.baseline.cycles;
    if (Number.isFinite(baselineCycles)) {
      lines.push(`Baseline schedule cycles: ${baselineCycles}`);
    }
    if (Number.isFinite(summary.baseline.utilization)) {
      lines.push(`Baseline utilization: ${formatScalar(summary.baseline.utilization, 4)}`);
    }
    if (Number.isFinite(summary.baseline.violations)) {
      lines.push(`Baseline violations: ${summary.baseline.violations}`);
    }
    if (Number.isFinite(datasetMeta?.baselineCycles) && Number.isFinite(baselineCycles)) {
      if (datasetMeta.baselineCycles !== baselineCycles) {
        lines.push(`Baseline mismatch: dataset=${datasetMeta.baselineCycles} schedule=${baselineCycles}`);
      }
    }
  }
  if (summary.specSearch) {
    const search = summary.specSearch;
    lines.push('Spec search (Layer 0):');
    lines.push(`  restarts ${search.restarts} • steps ${search.steps}`);
    lines.push(`  lambda ${formatScalar(search.cycleLambda, 3)} • gate ${formatScalar(search.penaltyGate, 3)} • fallback ${formatScalar(search.fallbackCycles, 0)}`);
    if (search.constraintMode) {
      lines.push(`  constraint mode ${search.constraintMode}`);
    }
    if (search.scheduler) {
      lines.push(`  scheduler ${search.scheduler}`);
    }
    if (search.scoreMode) {
      lines.push(`  score mode ${search.scoreMode}`);
    }
    if (Number.isFinite(search.lbPenalty) || Number.isFinite(search.targetCycles)) {
      lines.push(`  lb penalty ${formatScalar(search.lbPenalty, 3)} • target ${formatScalar(search.targetCycles, 0)}`);
    }
    lines.push(`  best cycles ${search.bestCycles} • penalty ${formatScalar(search.bestPenalty, 3)} • energy ${formatScalar(search.bestEnergy, 3)}`);
    if (search.bestSpecSignature) {
      lines.push(`  best spec ${search.bestSpecSignature}`);
    }
    if (Array.isArray(search.candidates) && search.candidates.length) {
      lines.push('  top specs:');
      search.candidates.forEach((candidate, index) => {
        const parts = [
          `#${index + 1}`,
          `cycles ${candidate.cycles}`,
          `pen ${formatScalar(candidate.penalty, 3)}`,
        ];
        if (candidate.signature) {
          parts.push(candidate.signature);
        }
        lines.push(`    ${parts.join(' • ')}`);
      });
    }
  }
  if (datasetMeta?.dependencyModel) {
    const model = datasetMeta.dependencyModel;
    lines.push('Dependency model:');
    lines.push(`  RAW=${!!model.includes_raw} WAW=${!!model.includes_waw} WAR=${!!model.includes_war} temp=${!!model.temp_hazard_tags} RAR=${!!model.read_after_read} latency=${model?.latency?.default ?? '--'}`);
  }
  if (datasetMeta?.dagHash) {
    lines.push(`DAG hash: ${datasetMeta.dagHash}`);
  }
  if (datasetMeta?.spec) {
    lines.push('Spec:');
    lines.push(...formatSpecLines(datasetMeta.spec));
  }
  if (Array.isArray(summary.candidates) && summary.candidates.length) {
    lines.push('Top candidates:');
    summary.candidates.forEach((candidate, index) => {
      const parts = [
        `#${index + 1}`,
        `restart ${candidate.restart}`,
        `cycles ${candidate.cycles}`,
        `util ${formatScalar(candidate.utilization, 4)}`,
        `viol ${candidate.violations}`,
        `steps ${candidate.steps}`,
      ];
      lines.push(`  ${parts.join(' • ')}`);
    });
  }
  summaryEl.textContent = lines.join('\n');
}

export function populateVliwBundleSelect(bundleCount) {
  const select = $('energy-vliw-bundle-select');
  if (!select) return;
  select.innerHTML = '';
  const allOption = document.createElement('option');
  allOption.value = '';
  allOption.textContent = 'All bundles';
  select.appendChild(allOption);
  if (!Number.isFinite(bundleCount) || bundleCount <= 0) return;
  for (let i = 0; i < bundleCount; i++) {
    const option = document.createElement('option');
    option.value = String(i);
    option.textContent = `Bundle ${i}`;
    select.appendChild(option);
  }
}

export function renderVliwBundleView(vliwState, selectedBundle) {
  const view = $('energy-bundle-view');
  if (!view) return;
  view.textContent = '';
  if (!vliwState || !vliwState.schedule) return;
  const { slotAssignments, slotEngines, slotIndices } = vliwState.schedule;
  if (!slotAssignments || !slotEngines || !slotIndices) return;
  const slotsPerCycle = slotEngines.length;
  if (!slotsPerCycle) return;
  const cycles = Math.floor(slotAssignments.length / slotsPerCycle);
  const lines = [];
  const showBundle = Number.isFinite(selectedBundle) ? selectedBundle : null;
  for (let cycle = 0; cycle < cycles; cycle++) {
    const parts = [`C${String(cycle).padStart(4, '0')}`];
    const baseIndex = cycle * slotsPerCycle;
    let hasBundle = false;
    for (let slot = 0; slot < slotsPerCycle; slot++) {
      const taskId = slotAssignments[baseIndex + slot];
      const engine = slotEngines[slot];
      const slotIndex = slotIndices[slot];
      if (taskId == null || taskId < 0) {
        parts.push(`${formatVliwSlotLabel(engine, slotIndex)}=--`);
        continue;
      }
      const meta = vliwState.taskMeta?.[taskId] || {};
      const bundle = Number.isFinite(meta.bundle) ? meta.bundle : '--';
      const deps = Number.isFinite(meta.deps) ? meta.deps : 0;
      const reads = Number.isFinite(meta.reads) ? meta.reads : 0;
      const writes = Number.isFinite(meta.writes) ? meta.writes : 0;
      const highlight = showBundle != null && bundle === showBundle;
      const prefix = highlight ? '*' : '';
      if (highlight) {
        hasBundle = true;
      }
      parts.push(
        `${prefix}${formatVliwSlotLabel(engine, slotIndex)}=${taskId}[b${bundle} d${deps} r${reads} w${writes}]`,
      );
    }
    if (showBundle != null && !hasBundle) {
      continue;
    }
    lines.push(parts.join(' '));
  }
  view.textContent = lines.join('\n');
}

export function drawEnergyChart(history = []) {
  const canvas = $('energy-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!history.length) return;
  const values = history.map((point) => point.energy ?? point.value ?? point);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  ctx.strokeStyle = '#4aa4ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = (index / Math.max(1, values.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

export function updateEnergyStats(result) {
  if (!result) {
    setText($('energy-stat-steps'), '--');
    setText($('energy-stat-energy'), '--');
    setText($('energy-stat-symmetry'), '--');
    setText($('energy-stat-count'), '--');
    setText($('energy-stat-binarize'), '--');
    setText($('energy-stat-dtype'), '--');
    setText($('energy-stat-backend'), '--');
    setText($('energy-stat-shape'), '--');
    setText($('energy-stat-mean'), '--');
    setText($('energy-stat-std'), '--');
    return;
  }
  const label = result.label || null;
  if (label) {
    setText($('energy-stat-steps'), label);
  } else {
    setText($('energy-stat-steps'), Number.isFinite(result.steps) ? String(result.steps) : '--');
  }
  setText($('energy-stat-energy'), Number.isFinite(result.energy) ? formatScalar(result.energy, 6) : '--');
  if (result.metrics) {
    setText($('energy-stat-symmetry'), formatScalar(result.metrics.cycles, 0));
    setText($('energy-stat-count'), formatScalar(result.metrics.utilization, 4));
    setText($('energy-stat-binarize'), formatScalar(result.metrics.violations, 0));
  } else {
    setText($('energy-stat-symmetry'), formatScalar(result.energyComponents?.symmetry, 6));
    setText($('energy-stat-count'), formatScalar(result.energyComponents?.count, 6));
    setText($('energy-stat-binarize'), formatScalar(result.energyComponents?.binarize, 6));
  }
  setText($('energy-stat-dtype'), result.dtype || '--');
  setText($('energy-stat-backend'), result.backend || '--');
  const shape = Array.isArray(result.shape) ? result.shape.join(' x ') : '--';
  setText($('energy-stat-shape'), shape);
  setText($('energy-stat-mean'), formatScalar(result.stateStats?.mean, 6));
  setText($('energy-stat-std'), formatScalar(result.stateStats?.std, 6));
}

export function clearEnergyChart() {
  const canvas = $('energy-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
