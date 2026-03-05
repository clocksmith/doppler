import { $, setText } from '../dom.js';
import { formatMs, formatScalar } from '../format.js';

function resolveEnergyGrid(shapeOrSize) {
  if (Array.isArray(shapeOrSize)) return shapeOrSize;
  if (Number.isFinite(shapeOrSize)) return [shapeOrSize, shapeOrSize];
  return [0, 0];
}

export function clearEnergyBoard() {
  const board = $('energy-board');
  if (!board) return;
  board.innerHTML = '';
  clearEnergyVector();
  clearEnergyIntensityBoard();
}

export function clearEnergyVector() {
  const vector = $('energy-vector');
  if (vector) vector.textContent = '';
}

export function clearEnergyIntensityBoard() {
  const board = $('energy-board-intensity');
  if (board) board.innerHTML = '';
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
  ctx.strokeStyle = '#000';
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
    setText($('energy-stat-time'), '--');
    setText($('energy-stat-steps'), '--');
    setText($('energy-stat-energy'), '--');
    setText($('energy-stat-symmetry'), '--');
    setText($('energy-stat-count'), '--');
    setText($('energy-stat-binarize'), '--');
    return;
  }

  setText(
    $('energy-stat-time'),
    Number.isFinite(result.totalTimeMs) ? formatMs(result.totalTimeMs) : '--',
  );
  setText(
    $('energy-stat-steps'),
    Number.isFinite(result.steps) ? String(result.steps) : '--',
  );
  setText(
    $('energy-stat-energy'),
    Number.isFinite(result.energy) ? formatScalar(result.energy, 6) : '--',
  );

  const components = result.energyComponents || {};
  setText(
    $('energy-stat-symmetry'),
    Number.isFinite(components.symmetry) ? formatScalar(components.symmetry, 6) : '--',
  );
  setText(
    $('energy-stat-count'),
    Number.isFinite(components.count) ? formatScalar(components.count, 6) : '--',
  );
  setText(
    $('energy-stat-binarize'),
    Number.isFinite(components.binarize) ? formatScalar(components.binarize, 6) : '--',
  );
}

export function clearEnergyChart() {
  const canvas = $('energy-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
