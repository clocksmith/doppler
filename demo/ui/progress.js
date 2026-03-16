import { $, setText } from './dom.js';
import { clampPercent } from './ui.js';

const PROGRESS_COMPLETE_DWELL_MS = 180;
const STORAGE_STAGE_DETAILS = Object.freeze({
  manifest: 'Reading local model files.',
  shards: 'Reading local model files.',
  tensors: 'Preparing model in memory.',
});
const GPU_STAGE_DETAILS = Object.freeze({
  layers: 'Preparing GPU resources.',
  gpu_transfer: 'Preparing GPU resources.',
  pipeline: 'Almost ready.',
  complete: 'Almost ready.',
});

export function showProgressOverlay(title, modelId) {
  const overlay = $('progress-overlay');
  const titleEl = $('progress-title');
  const modelEl = $('progress-model');
  const label = title || 'Preparing Local Model';
  if (titleEl) titleEl.textContent = label;
  if (modelEl) modelEl.textContent = modelId || '';
  if (overlay) delete overlay.dataset.completedAtMs;
  setProgressPhase('storage', 0, '--');
  setProgressPhase('gpu', 0, '--');
  setProgressDetail('Reading local model files.');
  setProgressShard(null, null);
  if (overlay) overlay.hidden = false;
}

export async function hideProgressOverlay() {
  const overlay = $('progress-overlay');
  if (!overlay) return;
  const completedAtMs = Number(overlay.dataset.completedAtMs);
  if (Number.isFinite(completedAtMs) && completedAtMs > 0) {
    const elapsedMs = Date.now() - completedAtMs;
    const remainingMs = PROGRESS_COMPLETE_DWELL_MS - elapsedMs;
    if (remainingMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, remainingMs));
    }
  }
  overlay.hidden = true;
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
  const normalizedProgress = Number.isFinite(info.progress)
    ? info.progress * 100
    : (Number.isFinite(info.percent) ? info.percent : 0);
  const percent = clampPercent(normalizedProgress);
  const label = `${Math.round(clampPercent(percent))}%`;
  const isGpuPhase = stage === 'layers' || stage === 'gpu_transfer' || stage === 'complete' || stage === 'pipeline';

  if (isGpuPhase) {
    setProgressPhase('storage', 100, '100%');
    setProgressPhase('gpu', percent, label);
    setProgressShard(null, null);
    if (stage === 'complete') {
      const overlay = $('progress-overlay');
      if (overlay) overlay.dataset.completedAtMs = String(Date.now());
    }
  } else {
    setProgressPhase('storage', percent, label);
    if (stage === 'shards' && info.shard != null && info.totalShards != null) {
      setProgressShard(info.shard, info.totalShards);
    }
  }

  setProgressDetail(
    STORAGE_STAGE_DETAILS[stage]
      || GPU_STAGE_DETAILS[stage]
      || info.message
      || ''
  );
}
