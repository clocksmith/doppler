import { $ } from './dom.js';
import { clampPercent } from './ui.js';

export function showProgressOverlay(title) {
  const overlay = $('progress-overlay');
  const titleEl = $('progress-title');
  if (titleEl && title) titleEl.textContent = title;
  setProgressPhase('source', 0, '--');
  setProgressPhase('gpu', 0, '--');
  if (overlay) overlay.hidden = false;
}

export function hideProgressOverlay() {
  const overlay = $('progress-overlay');
  if (overlay) overlay.hidden = true;
}

export function setProgressPhase(phase, percent, label) {
  const row = document.querySelector(`.progress-phase-row[data-phase="${phase}"]`);
  if (!row) return;
  const fill = row.querySelector('.progress-fill');
  const value = row.querySelector('.progress-phase-value');
  if (fill) fill.style.width = `${clampPercent(percent)}%`;
  if (value) value.textContent = label ?? `${Math.round(clampPercent(percent))}%`;
}

export function updateProgressFromLoader(info) {
  if (!info) return;
  const stage = info.stage || '';
  const percent = Number.isFinite(info.progress) ? info.progress * 100 : 0;
  const label = info.message || `${Math.round(percent)}%`;
  const phase = (stage === 'layers' || stage === 'gpu_transfer' || stage === 'complete' || stage === 'pipeline')
    ? 'gpu'
    : 'source';
  setProgressPhase(phase, percent, label);
  const titleEl = $('progress-title');
  if (titleEl && info.message) {
    titleEl.textContent = info.message;
  }
}
