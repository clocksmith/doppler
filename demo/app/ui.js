import { state } from './state.js';
import { $, setText } from './dom.js';

const STATUS_CLASSES = ['status-success', 'status-warning', 'status-error', 'status-info'];
const STATUS_MARKERS = {
  success: '●',
  warning: '▲',
  error: '☒',
  info: '○',
  default: '○',
};

export function showErrorModal(message) {
  const modal = $('error-modal');
  const messageEl = $('error-message');
  if (!modal || !messageEl) return;
  setText(messageEl, message ? String(message) : 'Unknown error');
  modal.hidden = false;
}

export function hideErrorModal() {
  const modal = $('error-modal');
  if (!modal) return;
  modal.hidden = true;
}

export function setStatusIndicator(message, tone) {
  const indicator = $('status-indicator');
  if (!indicator) return;
  const textEl = indicator.querySelector('.status-text');
  const marker = STATUS_MARKERS[tone] || STATUS_MARKERS.default;
  setText(textEl, `${marker} ${message}`);
  indicator.classList.remove(...STATUS_CLASSES);
  if (tone) {
    indicator.classList.add(`status-${tone}`);
  }
}

function formatMegabytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return '0.0';
  return (value / (1024 * 1024)).toFixed(1);
}

function formatDownloadStatus(progress) {
  if (!progress || typeof progress !== 'object') {
    return 'Downloading...';
  }
  const percent = Number(progress.percent);
  const downloadedBytes = Number(progress.downloadedBytes);
  const totalBytes = Number(progress.totalBytes);
  const percentLabel = Number.isFinite(percent) ? `${clampPercent(percent).toFixed(1)}%` : '';

  if (Number.isFinite(totalBytes) && totalBytes > 0) {
    const safeDownloaded = Number.isFinite(downloadedBytes) && downloadedBytes > 0 ? downloadedBytes : 0;
    const ratio = `${formatMegabytes(safeDownloaded)} / ${formatMegabytes(totalBytes)} MB`;
    return percentLabel ? `Downloading ${percentLabel} (${ratio})` : `Downloading (${ratio})`;
  }
  if (Number.isFinite(downloadedBytes) && downloadedBytes > 0) {
    const amount = `${formatMegabytes(downloadedBytes)} MB`;
    return percentLabel ? `Downloading ${percentLabel} (${amount})` : `Downloading (${amount})`;
  }
  return percentLabel ? `Downloading ${percentLabel}` : 'Downloading...';
}

export function updateStatusIndicator() {
  if (state.runLoading || state.diffusionLoading || state.energyLoading) {
    setStatusIndicator('Loading...', 'info');
    return;
  }
  if (state.convertActive) {
    setStatusIndicator('Converting...', 'info');
    return;
  }
  if (state.runGenerating && state.runPrefilling) {
    setStatusIndicator('Prefilling...', 'info');
    return;
  }
  if (state.runGenerating || state.diffusionGenerating || state.energyGenerating) {
    setStatusIndicator('Generating...', 'info');
    return;
  }
  if (state.downloadActive) {
    setStatusIndicator(formatDownloadStatus(state.downloadProgress), 'info');
    return;
  }
  setStatusIndicator('Ready', 'success');
}

export function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

export function setBarWidth(id, percent) {
  const el = $(id);
  if (!el) return;
  el.style.width = `${clampPercent(percent)}%`;
}
