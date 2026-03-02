import { $, setText } from './dom.js';
import { clampPercent } from './ui.js';

export function showProgressOverlay(title, modelId) {
  const overlay = $('progress-overlay');
  const titleEl = $('progress-title');
  const label = modelId ? `${title || 'Loading Model'} — ${modelId}` : (title || 'Loading Model');
  if (titleEl) titleEl.textContent = label;
  setProgressPhase('storage', 0, '--');
  setProgressPhase('gpu', 0, '--');
  setProgressDetail('');
  setProgressShard(null, null);
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

export function setProgressDetail(message) {
  setText($('progress-detail'), message || '');
}

export function setProgressShard(shard, totalShards) {
  const el = $('progress-shard');
  if (!el) return;
  if (shard != null && totalShards != null && totalShards > 0) {
    el.textContent = `Shard ${shard} / ${totalShards}`;
  } else {
    el.textContent = '';
  }
}

export function updateProgressFromLoader(info) {
  if (!info) return;
  const stage = info.stage || '';
  const percent = Number.isFinite(info.progress) ? info.progress * 100 : 0;
  const label = `${Math.round(clampPercent(percent))}%`;

  if (stage === 'layers' || stage === 'gpu_transfer' || stage === 'complete' || stage === 'pipeline') {
    setProgressPhase('gpu', percent, label);
    setProgressShard(null, null);
  } else {
    setProgressPhase('storage', percent, label);
    if (stage === 'shards' && info.shard != null && info.totalShards != null) {
      setProgressShard(info.shard, info.totalShards);
    }
  }

  setProgressDetail(info.message || '');
}
