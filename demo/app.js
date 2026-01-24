import { log } from '../src/debug/index.js';
import { listPresets, createConverterConfig } from '../src/config/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { listRegisteredModels } from '../src/storage/registry.js';
import { formatBytes } from '../src/storage/quota.js';
import {
  downloadModel,
  pauseDownload,
  resumeDownload,
  cancelDownload,
  listDownloads,
  formatSpeed,
  estimateTimeRemaining,
} from '../src/storage/downloader.js';
import {
  convertModel,
  createRemoteModelSources,
  isConversionSupported,
  pickModelFiles,
} from '../src/browser/browser-converter.js';
import { applyRuntimePreset } from '../src/inference/browser-harness.js';
import { DiagnosticsController } from './diagnostics-controller.js';

const MB = 1024 * 1024;

const state = {
  runtimeOverride: null,
  lastReport: null,
  lastReportInfo: null,
  conversionQueue: [],
  queueRunning: false,
  activeDownloadId: null,
};

const controller = new DiagnosticsController({ log });

function $(id) {
  return document.getElementById(id);
}

function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

function setHidden(el, hidden) {
  if (!el) return;
  el.hidden = Boolean(hidden);
}

function normalizeRuntimeConfig(raw) {
  if (!raw || typeof raw !== 'object') return null;
  if (raw.runtime && typeof raw.runtime === 'object') return raw.runtime;
  if (raw.shared || raw.loading || raw.inference || raw.emulation) return raw;
  return null;
}

function updateConvertStatus(message, percent) {
  const status = $('convert-status');
  const progress = $('convert-progress');
  const label = $('convert-message');
  if (!status || !progress || !label) return;
  setHidden(status, false);
  setText(label, message || '');
  if (Number.isFinite(percent)) {
    progress.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  }
}

function resetConvertStatus() {
  const status = $('convert-status');
  const progress = $('convert-progress');
  const label = $('convert-message');
  if (!status || !progress || !label) return;
  setHidden(status, true);
  progress.style.width = '0%';
  setText(label, 'Ready');
}

function renderConversionQueue() {
  const container = $('convert-queue');
  if (!container) return;
  if (!state.conversionQueue.length) {
    container.textContent = 'Queue empty';
    container.classList.add('muted');
    return;
  }
  container.classList.remove('muted');
  container.innerHTML = '';
  for (const item of state.conversionQueue) {
    const row = document.createElement('div');
    row.className = 'queue-item';
    row.dataset.state = item.state || 'pending';
    const label = document.createElement('span');
    label.className = 'queue-label';
    label.textContent = item.label;
    const status = document.createElement('span');
    status.className = 'queue-status';
    status.textContent = item.state || 'pending';
    row.appendChild(label);
    row.appendChild(status);
    container.appendChild(row);
  }
}

function enqueueConversion(item) {
  state.conversionQueue.push({
    ...item,
    state: 'pending',
  });
  renderConversionQueue();
}

function updateDiagnosticsStatus(message, isError = false) {
  const status = $('diagnostics-status');
  if (!status) return;
  status.textContent = message;
  status.dataset.state = isError ? 'error' : 'ready';
}

function updateDiagnosticsReport(text) {
  const report = $('diagnostics-report');
  if (!report) return;
  report.textContent = text;
}

async function refreshModelList() {
  const modelSelect = $('diagnostics-model');
  if (!modelSelect) return;
  modelSelect.innerHTML = '';
  let models = [];
  try {
    models = await listRegisteredModels();
  } catch (error) {
    log.warn('DopplerDemo', `Model registry unavailable: ${error.message}`);
  }
  if (!models.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No local models';
    modelSelect.appendChild(opt);
    return;
  }
  for (const model of models) {
    const opt = document.createElement('option');
    opt.value = model.modelId || model.id || '';
    opt.textContent = model.modelId || model.id || 'unknown';
    modelSelect.appendChild(opt);
  }
}

function populateModelPresets() {
  const presetSelect = $('convert-model-preset');
  if (!presetSelect) return;
  presetSelect.innerHTML = '';
  const autoOpt = document.createElement('option');
  autoOpt.value = '';
  autoOpt.textContent = 'auto';
  presetSelect.appendChild(autoOpt);
  for (const presetId of listPresets()) {
    const opt = document.createElement('option');
    opt.value = presetId;
    opt.textContent = presetId;
    presetSelect.appendChild(opt);
  }
}

function buildConverterConfig() {
  const allowDownload = $('convert-allow-download');
  const maxDownload = $('convert-max-download');
  const presetSelect = $('convert-model-preset');
  const presetId = presetSelect?.value?.trim() || null;
  const maxDownloadMb = Number(maxDownload?.value || 0);

  const config = createConverterConfig();
  if (presetId) {
    config.presets.model = presetId;
  }
  if (allowDownload) {
    config.http.allowDownloadFallback = allowDownload.checked;
  }
  if (Number.isFinite(maxDownloadMb) && maxDownloadMb > 0) {
    config.http.maxDownloadBytes = Math.round(maxDownloadMb * MB);
  }
  return config;
}

async function runConversion(files, converterConfig, label) {
  if (!isConversionSupported()) {
    throw new Error('Browser conversion requires OPFS or IndexedDB.');
  }
  updateConvertStatus(`Preparing conversion${label ? ` (${label})` : ''}...`, 0);
  const modelId = await convertModel(files, {
    converterConfig,
    onProgress: (update) => {
      if (!update) return;
      const percent = Number.isFinite(update.percent) ? update.percent : null;
      const message = update.message || 'Converting...';
      updateConvertStatus(label ? `${message} (${label})` : message, percent);
    },
  });
  updateConvertStatus(`Conversion complete: ${modelId}`, 100);
  await refreshModelList();
}

async function handleConvertFiles() {
  const files = await pickModelFiles();
  if (!files || files.length === 0) return;
  const converterConfig = buildConverterConfig();
  await runConversion(files, converterConfig);
}

async function handleConvertUrls() {
  const urlInput = $('convert-url-input');
  if (!urlInput) return;
  const urls = urlInput.value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (!urls.length) return;
  const converterConfig = buildConverterConfig();
  const sources = await createRemoteModelSources(urls, { converterConfig });
  await runConversion(sources, converterConfig);
}

async function handleQueueFiles() {
  const files = await pickModelFiles();
  if (!files || files.length === 0) return;
  const names = files.map((file) => file.name).filter(Boolean);
  enqueueConversion({
    type: 'files',
    files,
    label: names.length > 0 ? names.slice(0, 3).join(', ') : 'Local files',
  });
}

async function handleQueueUrls() {
  const urlInput = $('convert-url-input');
  if (!urlInput) return;
  const urls = urlInput.value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (!urls.length) return;
  enqueueConversion({
    type: 'urls',
    urls,
    label: urls.length === 1 ? urls[0] : `${urls.length} URLs`,
  });
}

async function runConversionQueue() {
  if (state.queueRunning) return;
  state.queueRunning = true;
  renderConversionQueue();
  for (const item of state.conversionQueue) {
    if (item.state !== 'pending') continue;
    item.state = 'running';
    renderConversionQueue();
    try {
      const converterConfig = buildConverterConfig();
      if (item.type === 'files') {
        await runConversion(item.files, converterConfig, item.label);
      } else {
        const sources = await createRemoteModelSources(item.urls, { converterConfig });
        await runConversion(sources, converterConfig, item.label);
      }
      item.state = 'done';
    } catch (error) {
      item.state = 'error';
      updateConvertStatus(`Conversion error: ${error.message}`);
    }
    renderConversionQueue();
  }
  state.queueRunning = false;
}

function clearConversionQueue() {
  state.conversionQueue = [];
  renderConversionQueue();
}

function updateDownloadStatus(progress) {
  const status = $('download-status');
  const bar = $('download-progress');
  const label = $('download-message');
  if (!status || !bar || !label) return;
  if (!progress) {
    setHidden(status, true);
    bar.style.width = '0%';
    setText(label, 'Idle');
    return;
  }
  setHidden(status, false);
  const percent = Number.isFinite(progress.percent) ? progress.percent : 0;
  bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  const remaining = Math.max(0, (progress.totalBytes || 0) - (progress.downloadedBytes || 0));
  const speed = Number(progress.speed || 0);
  const eta = speed > 0 ? estimateTimeRemaining(remaining, speed) : 'Calculating...';
  const detail = `${formatBytes(progress.downloadedBytes || 0)} / ${formatBytes(progress.totalBytes || 0)}`;
  const speedLabel = speed > 0 ? formatSpeed(speed) : '--';
  setText(label, `${progress.status || 'downloading'} • ${percent.toFixed(1)}% • ${detail} • ${speedLabel} • ETA ${eta}`);
}

function renderDownloadList(downloads) {
  const container = $('download-list');
  if (!container) return;
  container.innerHTML = '';
  if (!downloads || downloads.length === 0) {
    container.textContent = 'No downloads tracked';
    container.classList.add('muted');
    return;
  }
  container.classList.remove('muted');
  for (const entry of downloads) {
    const row = document.createElement('div');
    row.className = 'download-row';
    const name = document.createElement('span');
    name.textContent = entry.modelId || 'unknown';
    const stats = document.createElement('span');
    const percent = Number.isFinite(entry.percent) ? entry.percent : 0;
    stats.textContent = `${percent.toFixed(1)}% • ${entry.status || 'idle'}`;
    row.appendChild(name);
    row.appendChild(stats);
    container.appendChild(row);
  }
}

async function refreshDownloads() {
  try {
    const downloads = await listDownloads();
    renderDownloadList(downloads);
  } catch (error) {
    renderDownloadList([]);
    updateDownloadStatus({ status: `Error: ${error.message}`, percent: 0, downloadedBytes: 0, totalBytes: 0 });
  }
}

async function startDownload() {
  const baseUrl = $('download-base-url')?.value?.trim();
  if (!baseUrl) {
    updateDownloadStatus({ status: 'Missing base URL', percent: 0, downloadedBytes: 0, totalBytes: 0 });
    return;
  }
  const modelIdOverride = $('download-model-id')?.value?.trim() || undefined;
  updateDownloadStatus({ status: 'Starting...', percent: 0, downloadedBytes: 0, totalBytes: 0 });
  try {
    await downloadModel(baseUrl, (progress) => {
      if (!progress) return;
      state.activeDownloadId = progress.modelId || modelIdOverride || null;
      updateDownloadStatus(progress);
    }, { modelId: modelIdOverride });
    updateDownloadStatus({ status: 'Complete', percent: 100, downloadedBytes: 0, totalBytes: 0 });
    await refreshModelList();
    await refreshDownloads();
  } catch (error) {
    updateDownloadStatus({ status: `Error: ${error.message}`, percent: 0, downloadedBytes: 0, totalBytes: 0 });
  }
}

async function pauseActiveDownload() {
  const modelId = state.activeDownloadId || $('download-model-id')?.value?.trim();
  if (!modelId) return;
  pauseDownload(modelId);
  await refreshDownloads();
}

async function resumeActiveDownload() {
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
  }
}

async function cancelActiveDownload() {
  const modelId = state.activeDownloadId || $('download-model-id')?.value?.trim();
  if (!modelId) return;
  await cancelDownload(modelId);
  await refreshDownloads();
  updateDownloadStatus(null);
}

function handleRuntimeConfigFile(file) {
  if (!file) return;
  file.text()
    .then((text) => JSON.parse(text))
    .then((json) => {
      const runtime = normalizeRuntimeConfig(json);
      if (!runtime) {
        throw new Error('Runtime config file is missing runtime fields');
      }
      state.runtimeOverride = runtime;
      setRuntimeConfig(runtime);
      setText($('runtime-config-status'), `Override: ${file.name}`);
    })
    .catch((error) => {
      updateDiagnosticsStatus(`Runtime config error: ${error.message}`, true);
    });
}

async function applySelectedRuntimePreset() {
  const presetSelect = $('runtime-preset');
  if (!presetSelect) return;
  const presetId = presetSelect.value || 'default';
  try {
    await applyRuntimePreset(presetId);
    if (!state.runtimeOverride) {
      setText($('runtime-config-status'), `Preset: ${presetId}`);
    }
  } catch (error) {
    updateDiagnosticsStatus(`Preset error: ${error.message}`, true);
  }
}

async function handleDiagnosticsRun(mode) {
  const suiteSelect = $('diagnostics-suite');
  const modelSelect = $('diagnostics-model');
  const presetSelect = $('runtime-preset');
  const suite = suiteSelect?.value || 'inference';
  const modelId = modelSelect?.value || null;
  const runtimePreset = presetSelect?.value || 'default';

  updateDiagnosticsStatus(`${mode === 'verify' ? 'Verifying' : 'Running'} ${suite}...`);
  try {
    if (mode === 'verify') {
      await controller.verifySuite(null, {
        suite,
        runtimeConfig: state.runtimeOverride || getRuntimeConfig(),
      });
      updateDiagnosticsStatus('Verified');
      return;
    }

    const options = {
      suite,
      runtimePreset,
      modelId,
    };
    if (state.runtimeOverride) {
      options.runtimeConfig = state.runtimeOverride;
    }
    const result = await controller.runSuite(
      modelId ? { sources: { browser: { id: modelId } } } : null,
      options
    );
    state.lastReport = result.report;
    state.lastReportInfo = result.reportInfo;
    updateDiagnosticsStatus(`Complete (${result.suite})`);
    if (result.reportInfo?.path) {
      updateDiagnosticsReport(result.reportInfo.path);
    } else if (result.report?.timestamp) {
      updateDiagnosticsReport(result.report.timestamp);
    }
  } catch (error) {
    updateDiagnosticsStatus(error.message, true);
  }
}

function exportDiagnosticsReport() {
  if (!state.lastReport) {
    updateDiagnosticsStatus('No report available to export', true);
    return;
  }
  const timestamp = state.lastReport.timestamp || new Date().toISOString();
  const safeTimestamp = timestamp.replace(/[:]/g, '-');
  const filename = `doppler-report-${safeTimestamp}.json`;
  const blob = new Blob([JSON.stringify(state.lastReport, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bindUI() {
  const convertBtn = $('convert-btn');
  const convertUrlBtn = $('convert-url-btn');
  const convertQueueFiles = $('convert-queue-files');
  const convertQueueUrls = $('convert-queue-urls');
  const convertQueueRun = $('convert-queue-run');
  const convertQueueClear = $('convert-queue-clear');
  const downloadStart = $('download-start-btn');
  const downloadPause = $('download-pause-btn');
  const downloadResume = $('download-resume-btn');
  const downloadCancel = $('download-cancel-btn');
  const downloadRefresh = $('download-refresh-btn');
  const runtimePreset = $('runtime-preset');
  const runtimeFile = $('runtime-config-file');
  const runtimeClear = $('runtime-config-clear');
  const diagnosticsRun = $('diagnostics-run-btn');
  const diagnosticsVerify = $('diagnostics-verify-btn');
  const diagnosticsExport = $('diagnostics-export-btn');

  convertBtn?.addEventListener('click', () => {
    resetConvertStatus();
    handleConvertFiles().catch((error) => {
      updateConvertStatus(`Conversion error: ${error.message}`);
    });
  });

  convertUrlBtn?.addEventListener('click', () => {
    resetConvertStatus();
    handleConvertUrls().catch((error) => {
      updateConvertStatus(`Conversion error: ${error.message}`);
    });
  });

  convertQueueFiles?.addEventListener('click', () => {
    handleQueueFiles().catch((error) => {
      updateConvertStatus(`Queue error: ${error.message}`);
    });
  });

  convertQueueUrls?.addEventListener('click', () => {
    handleQueueUrls().catch((error) => {
      updateConvertStatus(`Queue error: ${error.message}`);
    });
  });

  convertQueueRun?.addEventListener('click', () => {
    resetConvertStatus();
    runConversionQueue().catch((error) => {
      updateConvertStatus(`Queue error: ${error.message}`);
    });
  });

  convertQueueClear?.addEventListener('click', clearConversionQueue);

  downloadStart?.addEventListener('click', () => {
    startDownload();
  });

  downloadPause?.addEventListener('click', () => {
    pauseActiveDownload();
  });

  downloadResume?.addEventListener('click', () => {
    resumeActiveDownload();
  });

  downloadCancel?.addEventListener('click', () => {
    cancelActiveDownload();
  });

  downloadRefresh?.addEventListener('click', () => {
    refreshDownloads();
  });

  runtimePreset?.addEventListener('change', () => {
    state.runtimeOverride = null;
    applySelectedRuntimePreset();
  });

  runtimeFile?.addEventListener('change', () => {
    const file = runtimeFile.files?.[0] || null;
    handleRuntimeConfigFile(file);
  });

  runtimeClear?.addEventListener('click', () => {
    state.runtimeOverride = null;
    runtimeFile.value = '';
    applySelectedRuntimePreset();
  });

  diagnosticsRun?.addEventListener('click', () => handleDiagnosticsRun('run'));
  diagnosticsVerify?.addEventListener('click', () => handleDiagnosticsRun('verify'));
  diagnosticsExport?.addEventListener('click', exportDiagnosticsReport);
}

async function init() {
  populateModelPresets();
  await refreshModelList();
  await applySelectedRuntimePreset();
  renderConversionQueue();
  await refreshDownloads();
  bindUI();
}

init().catch((error) => {
  log.error('DopplerDemo', `Demo init failed: ${error.message}`);
});
