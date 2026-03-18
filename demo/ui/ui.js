import { state } from './state.js';
import { $, setText, setHidden } from './dom.js';

const STATUS_CLASSES = ['status-success', 'status-warning', 'status-error', 'status-info'];
const STATUS_MARKERS = {
  success: '★',
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
  const markerEl = indicator.querySelector('.status-marker');
  if (markerEl) {
    setText(markerEl, marker);
  }
  indicator.classList.remove(...STATUS_CLASSES);
  if (tone) {
    indicator.classList.add(`status-${tone}`);
  }
  if (!textEl) return;
  setText(textEl, message || '');
}

function formatMegabytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return '0.0';
  return (value / (1024 * 1024)).toFixed(1);
}

function formatDownloadSpeed(bytesPerSecond) {
  const value = Number(bytesPerSecond);
  if (!Number.isFinite(value) || value <= 0) return '';
  return `${(value / (1024 * 1024)).toFixed(1)} MB/s`;
}

function joinDownloadStatusParts(parts) {
  const compact = parts.filter((part) => typeof part === 'string' && part.trim().length > 0);
  return compact.length > 0 ? ` (${compact.join(' · ')})` : '';
}

function formatDownloadStatus(progress) {
  if (!progress || typeof progress !== 'object') {
    return 'Downloading...';
  }
  const percent = Number(progress.percent);
  const downloadedBytes = Number(progress.downloadedBytes);
  const totalBytes = Number(progress.totalBytes);
  const speedBytesPerSecond = Number(progress.speed);
  const totalShards = Number(progress.totalShards);
  const completedShards = Number(progress.completedShards);
  const currentShard = Number(progress.currentShard);
  const percentLabel = Number.isFinite(percent) ? `${clampPercent(percent).toFixed(1)}%` : '';
  const shardLabel = Number.isFinite(totalShards) && totalShards > 0 && Number.isFinite(completedShards)
    ? `shards ${Math.max(0, completedShards)}/${Math.max(0, totalShards)}`
    : '';
  const currentShardLabel = Number.isFinite(currentShard) && currentShard > 0 && Number.isFinite(totalShards) && totalShards > 0
    ? `shard ${Math.round(currentShard)}/${Math.max(0, totalShards)}`
    : '';
  const speedLabel = formatDownloadSpeed(speedBytesPerSecond);

  if (Number.isFinite(totalBytes) && totalBytes > 0) {
    const safeDownloaded = Number.isFinite(downloadedBytes) && downloadedBytes > 0 ? downloadedBytes : 0;
    const ratio = `${formatMegabytes(safeDownloaded)} / ${formatMegabytes(totalBytes)} MB`;
    const details = joinDownloadStatusParts([currentShardLabel, shardLabel, speedLabel, ratio]);
    return percentLabel ? `Downloading ${percentLabel}${details}` : `Downloading${details}`;
  }
  if (Number.isFinite(downloadedBytes) && downloadedBytes > 0) {
    const amount = `${formatMegabytes(downloadedBytes)} MB`;
    const details = joinDownloadStatusParts([currentShardLabel, shardLabel, speedLabel, amount]);
    return percentLabel ? `Downloading ${percentLabel}${details}` : `Downloading${details}`;
  }
  const details = joinDownloadStatusParts([currentShardLabel, shardLabel, speedLabel]);
  if (percentLabel) {
    return `Downloading ${percentLabel}${details}`;
  }
  return details ? `Downloading${details}` : 'Downloading...';
}

export function updateStatusIndicator() {
  if (state.appInitializing) {
    setStatusIndicator('Initializing', 'info');
    return;
  }
  if (state.modelAvailabilityLoading) {
    setStatusIndicator('Loading...', 'info');
    return;
  }
  if (state.runLoading || state.compareLoading || state.diffusionLoading || state.energyLoading) {
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
  if (state.compareGenerating) {
    setStatusIndicator('Comparing...', 'info');
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
