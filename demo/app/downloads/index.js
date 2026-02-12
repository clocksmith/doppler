import { formatBytes } from '../../../src/storage/quota.js';
import {
  downloadModel,
  pauseDownload,
  resumeDownload,
  cancelDownload,
  listDownloads,
  formatSpeed,
  estimateTimeRemaining,
} from '../../../src/storage/downloader.js';
import { state } from '../state.js';
import { $, setHidden, setText } from '../dom.js';
import { updateStatusIndicator, showErrorModal } from '../ui.js';

let callbacks = {
  onModelRegistered: null,
  onModelsUpdated: null,
};

export function configureDownloadCallbacks(nextCallbacks = {}) {
  callbacks = {
    ...callbacks,
    ...nextCallbacks,
  };
}

export function updateDownloadStatus(progress) {
  const status = $('download-status');
  const bar = $('download-progress');
  const label = $('download-message');
  if (!status || !bar || !label) return;
  if (!progress) {
    setHidden(status, true);
    bar.style.width = '0%';
    setText(label, 'Idle');
    state.downloadActive = false;
    updateStatusIndicator();
    return;
  }
  setHidden(status, false);
  const percent = Number.isFinite(progress.percent) ? progress.percent : 0;
  const statusText = String(progress.status || '');
  const lowered = statusText.toLowerCase();
  const isComplete = lowered.includes('complete');
  const isPaused = lowered.includes('pause');
  const isError = lowered.includes('error');
  const isIdle = lowered.includes('idle');
  state.downloadActive = !isComplete
    && !isPaused
    && !isError
    && !isIdle
    && (lowered.includes('download') || lowered.includes('start') || (percent > 0 && percent < 100));
  updateStatusIndicator();
  bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  const remaining = Math.max(0, (progress.totalBytes || 0) - (progress.downloadedBytes || 0));
  const speed = Number(progress.speed || 0);
  const eta = speed > 0 ? estimateTimeRemaining(remaining, speed) : 'Calculating...';
  const detail = `${formatBytes(progress.downloadedBytes || 0)} / ${formatBytes(progress.totalBytes || 0)}`;
  const speedLabel = speed > 0 ? formatSpeed(speed) : '--';
  setText(label, `${progress.status || 'downloading'} - ${percent.toFixed(1)}% - ${detail} - ${speedLabel} - ETA ${eta}`);
}

function renderDownloadList(downloads) {
  const container = $('download-list');
  if (!container) return;
  container.innerHTML = '';
  if (!downloads || downloads.length === 0) {
    container.textContent = 'No downloads tracked';
    return;
  }
  for (const entry of downloads) {
    const row = document.createElement('div');
    row.className = 'download-row';
    const name = document.createElement('span');
    name.textContent = entry.modelId || 'unknown';
    const stats = document.createElement('span');
    const percent = Number.isFinite(entry.percent) ? entry.percent : 0;
    stats.textContent = `${percent.toFixed(1)}% - ${entry.status || 'idle'}`;
    row.appendChild(name);
    row.appendChild(stats);
    container.appendChild(row);
  }
}

export async function refreshDownloads() {
  try {
    const downloads = await listDownloads();
    renderDownloadList(downloads);
    const active = downloads.some((entry) => String(entry.status || '').toLowerCase() === 'downloading');
    state.downloadActive = active;
    updateStatusIndicator();
  } catch (error) {
    renderDownloadList([]);
    updateDownloadStatus({ status: `Error: ${error.message}`, percent: 0, downloadedBytes: 0, totalBytes: 0 });
  }
}

async function runDownloadFromBaseUrl(baseUrl, modelIdOverride, { showErrorModalOnFailure = true } = {}) {
  const normalizedBaseUrl = String(baseUrl || '').trim();
  if (!normalizedBaseUrl) {
    const error = new Error('Missing base URL');
    updateDownloadStatus({ status: error.message, percent: 0, downloadedBytes: 0, totalBytes: 0 });
    throw error;
  }

  const normalizedModelId =
    typeof modelIdOverride === 'string' && modelIdOverride.trim()
      ? modelIdOverride.trim()
      : undefined;

  let downloadedModelId = normalizedModelId ?? null;
  updateDownloadStatus({ status: 'Starting...', percent: 0, downloadedBytes: 0, totalBytes: 0 });

  try {
    await downloadModel(
      normalizedBaseUrl,
      (progress) => {
        if (!progress) return;
        state.activeDownloadId = progress.modelId || normalizedModelId || null;
        downloadedModelId = progress.modelId || downloadedModelId;
        updateDownloadStatus(progress);
        callbacks.onProgress?.(progress);
      },
      { modelId: normalizedModelId }
    );

    if (downloadedModelId && callbacks.onModelRegistered) {
      await callbacks.onModelRegistered(downloadedModelId);
    }

    updateDownloadStatus({ status: 'Complete', percent: 100, downloadedBytes: 0, totalBytes: 0 });
    callbacks.onStateChange?.({ active: false, modelId: downloadedModelId });

    if (callbacks.onModelsUpdated) {
      await callbacks.onModelsUpdated();
    }

    await refreshDownloads();
    return downloadedModelId;
  } catch (error) {
    updateDownloadStatus({ status: `Error: ${error.message}`, percent: 0, downloadedBytes: 0, totalBytes: 0 });
    callbacks.onStateChange?.({ active: false, modelId: downloadedModelId ?? null });
    if (showErrorModalOnFailure) {
      showErrorModal(`Download failed: ${error.message}`);
    }
    throw error;
  }
}

export async function startDownload() {
  const baseUrl = $('download-base-url')?.value?.trim();
  if (!baseUrl) {
    updateDownloadStatus({ status: 'Missing base URL', percent: 0, downloadedBytes: 0, totalBytes: 0 });
    return;
  }
  const modelIdOverride = $('download-model-id')?.value?.trim() || undefined;
  try {
    await runDownloadFromBaseUrl(baseUrl, modelIdOverride, { showErrorModalOnFailure: true });
  } catch {
    // Error UI is handled in runDownloadFromBaseUrl for manual downloads.
  }
}

export async function startDownloadFromBaseUrl(baseUrl, modelIdOverride) {
  return runDownloadFromBaseUrl(baseUrl, modelIdOverride, { showErrorModalOnFailure: false });
}

export async function pauseActiveDownload() {
  const modelId = state.activeDownloadId || $('download-model-id')?.value?.trim();
  if (!modelId) return;
  try {
    pauseDownload(modelId);
    await refreshDownloads();
  } catch (error) {
    updateDownloadStatus({ status: `Error: ${error.message}`, percent: 0, downloadedBytes: 0, totalBytes: 0 });
    showErrorModal(`Pause failed: ${error.message}`);
  }
}

export async function resumeActiveDownload() {
  const modelId = state.activeDownloadId || $('download-model-id')?.value?.trim();
  if (!modelId) return;
  try {
    await resumeDownload(modelId, (progress) => {
      if (!progress) return;
      state.activeDownloadId = progress.modelId || modelId;
      updateDownloadStatus(progress);
    });
    await refreshDownloads();
  } catch (error) {
    updateDownloadStatus({ status: `Error: ${error.message}`, percent: 0, downloadedBytes: 0, totalBytes: 0 });
    showErrorModal(`Resume failed: ${error.message}`);
  }
}

export async function cancelActiveDownload() {
  const modelId = state.activeDownloadId || $('download-model-id')?.value?.trim();
  if (!modelId) return;
  try {
    await cancelDownload(modelId);
    await refreshDownloads();
    updateDownloadStatus(null);
  } catch (error) {
    updateDownloadStatus({ status: `Error: ${error.message}`, percent: 0, downloadedBytes: 0, totalBytes: 0 });
    showErrorModal(`Cancel failed: ${error.message}`);
  }
}
