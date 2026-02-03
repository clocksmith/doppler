import { log } from '../src/debug/index.js';
import { listPresets, createConverterConfig, detectPreset, resolvePreset } from '../src/config/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../src/config/schema/index.js';
import { listRegisteredModels, removeRegisteredModel } from '../src/storage/registry.js';
import { formatBytes, getQuotaInfo } from '../src/storage/quota.js';
import {
  openModelStore,
  loadManifestFromStore,
  loadShard,
  loadTensorsFromStore,
  saveManifest,
  saveTensorsToStore,
  loadTokenizerFromStore,
  loadTokenizerModelFromStore,
  deleteModel,
} from '../src/storage/shard-manager.js';
import { parseManifest, getManifest, setManifest, clearManifest, classifyTensorRole } from '../src/storage/rdrr-format.js';
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
} from '../src/browser/browser-converter.js';
import { buildManifestInference, inferEmbeddingOutputConfig } from '../src/converter/manifest-inference.js';
import { pickModelDirectory, pickModelFiles } from '../src/browser/file-picker.js';
import { applyRuntimePreset, loadRuntimePreset } from '../src/inference/browser-harness.js';
import { createPipeline } from '../src/inference/pipeline.js';
import { formatChatMessages } from '../src/inference/pipeline/chat-format.js';
import { initDevice, getDevice, getKernelCapabilities, getPlatformConfig, isWebGPUAvailable } from '../src/gpu/device.js';
import { captureMemorySnapshot } from '../src/loader/memory-monitor.js';
import { destroyBufferPool } from '../src/memory/buffer-pool.js';
import { DiagnosticsController } from './diagnostics-controller.js';

const state = {
  runtimeOverride: null,
  runtimeOverrideBase: null,
  runtimeOverrideLabel: null,
  runtimeSamplingOverride: null,
  runtimeSamplingLabel: null,
  samplingSyncing: false,
  lastReport: null,
  lastReportInfo: null,
  lastMetrics: null,
  lastInferenceStats: null,
  lastMemoryStats: null,
  activePipeline: null,
  activePipelineModelId: null,
  activeModelId: null,
  modelTypeCache: {},
  modeModelId: { run: null, diffusion: null },
  chatMessages: [],
  chatAbortController: null,
  chatGenerating: false,
  chatLoading: false,
  diffusionGenerating: false,
  diffusionLoading: false,
  convertActive: false,
  downloadActive: false,
  uiMode: 'run',
  runLog: [],
  runCounter: 0,
  storageUsageBytes: 0,
  storageQuotaBytes: 0,
  storageInspectorScanning: false,
  storageInspectorLastScan: 0,
  gpuMaxBytes: 0,
  systemMemoryBytes: 0,
  uiIntervalId: null,
  lastStorageRefresh: 0,
  activeDownloadId: null,
};

const RUNTIME_CONFIG_PRESETS = [
  { id: '', label: 'none' },
  { id: 'modes/production', label: 'modes/production' },
  { id: 'modes/low-memory', label: 'modes/low-memory' },
  { id: 'modes/simulation', label: 'modes/simulation' },
  { id: 'modes/trace-layers', label: 'modes/trace-layers' },
  { id: 'kernels/safe-q4k', label: 'kernels/safe-q4k' },
  { id: 'kernels/fused-q4k', label: 'kernels/fused-q4k' },
  { id: 'kernels/dequant-f16-q4k', label: 'kernels/dequant-f16-q4k' },
  { id: 'kernels/dequant-f32-q4k', label: 'kernels/dequant-f32-q4k' },
  { id: 'compute/f16-activations', label: 'compute/f16-activations' },
  { id: 'compute/f16-batched', label: 'compute/f16-batched' },
  { id: 'platform/metal-apple-q4k', label: 'platform/metal-apple-q4k' },
  { id: 'model/gemma3-layer-probe', label: 'model/gemma3-layer-probe' },
  { id: 'model/gemma2-pipeline', label: 'model/gemma2-pipeline' },
  { id: 'model/gemma2-pipeline-debug', label: 'model/gemma2-pipeline-debug' },
  { id: 'experiments/gemma3-verify', label: 'experiments/gemma3-verify' },
  { id: 'experiments/gemma3-debug-q4k', label: 'experiments/gemma3-debug-q4k' },
];

const controller = new DiagnosticsController({ log });

function $(id) {
  return document.getElementById(id);
}

function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

const STATUS_CLASSES = ['status-success', 'status-warning', 'status-error', 'status-info'];

function setStatusIndicator(message, tone) {
  const indicator = $('status-indicator');
  if (!indicator) return;
  const textEl = indicator.querySelector('.status-text');
  const dot = indicator.querySelector('.status-dot');
  setText(textEl, message);
  indicator.classList.remove(...STATUS_CLASSES);
  if (tone) {
    indicator.classList.add(`status-${tone}`);
  }
  if (dot) {
    if (tone) {
      dot.classList.add('status-dot-filled');
    } else {
      dot.classList.remove('status-dot-filled');
    }
  }
}

function updateStatusIndicator() {
  if (state.chatLoading) {
    setStatusIndicator('Loading model', 'info');
    return;
  }
  if (state.diffusionLoading) {
    setStatusIndicator('Loading diffusion', 'info');
    return;
  }
  if (state.convertActive) {
    setStatusIndicator('Converting', 'info');
    return;
  }
  if (state.chatGenerating) {
    setStatusIndicator('Generating', 'info');
    return;
  }
  if (state.diffusionGenerating) {
    setStatusIndicator('Generating', 'info');
    return;
  }
  if (state.downloadActive) {
    setStatusIndicator('Downloading', 'info');
    return;
  }
  setStatusIndicator('Ready', 'success');
}

function applyModeVisibility(mode) {
  const panels = document.querySelectorAll('[data-modes]');
  panels.forEach((panel) => {
    const modes = panel.dataset.modes?.split(/\s+/).filter(Boolean) || [];
    panel.hidden = modes.length > 0 && !modes.includes(mode);
  });
}

function setUiMode(mode) {
  const app = $('app');
  if (!app) return;
  state.uiMode = mode;
  app.dataset.mode = mode;
  document.querySelectorAll('.mode-tab').forEach((button) => {
    const isActive = button.dataset.mode === mode;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
  applyModeVisibility(mode);
  if (mode === 'models') {
    refreshStorageInspector();
  }
  syncModelForMode(mode).catch((error) => {
    log.warn('DopplerDemo', `Mode model sync failed: ${error.message}`);
  });
}

function setHidden(el, hidden) {
  if (!el) return;
  el.hidden = Boolean(hidden);
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function setBarWidth(id, percent) {
  const el = $(id);
  if (!el) return;
  el.style.width = `${clampPercent(percent)}%`;
}

function formatRate(value) {
  if (!Number.isFinite(value)) return '--';
  return `${value.toFixed(2)} tok/s`;
}

function formatMs(value) {
  if (!Number.isFinite(value)) return '--';
  return `${Math.round(value)}ms`;
}

function normalizeRuntimeConfig(raw) {
  if (!raw || typeof raw !== 'object') return null;
  if (raw.runtime && typeof raw.runtime === 'object') return raw.runtime;
  if (raw.shared || raw.loading || raw.inference || raw.emulation) return raw;
  return null;
}

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function mergeRuntimeOverrides(base, override) {
  if (override === undefined) return base;
  if (override === null) return null;
  if (!isPlainObject(base) || !isPlainObject(override)) {
    return override;
  }
  const merged = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (value === undefined) continue;
    merged[key] = mergeRuntimeOverrides(base[key], value);
  }
  return merged;
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
  setHidden(status, false);
  progress.style.width = '0%';
  setText(label, 'Ready');
}

function updateDiffusionStatus(message) {
  const status = $('diffusion-output-status');
  if (!status) return;
  setText(status, message || 'Idle');
}

function updateExportStatus(message, percent) {
  const status = $('export-status');
  const progress = $('export-progress');
  const label = $('export-message');
  if (!status || !progress || !label) return;
  setHidden(status, false);
  setText(label, message || '');
  if (Number.isFinite(percent)) {
    progress.style.width = `${clampPercent(percent)}%`;
  }
}

function resetExportStatus() {
  const status = $('export-status');
  const progress = $('export-progress');
  const label = $('export-message');
  if (!status || !progress || !label) return;
  setHidden(status, false);
  progress.style.width = '0%';
  setText(label, 'Idle');
}

async function deriveModelIdFromFiles(files, fallbackLabel) {
  const fallback = fallbackLabel?.trim();
  if (fallback) return fallback;

  const configFile = files.find((file) => file.name === 'config.json');
  if (configFile) {
    try {
      const text = await configFile.text();
      const json = JSON.parse(text);
      const rawName = json?._name_or_path || json?.model_id || json?.modelId || json?.name;
      if (typeof rawName === 'string' && rawName.trim()) {
        const parts = rawName.trim().split('/');
        const name = parts[parts.length - 1];
        if (name) return name;
      }
    } catch {
      // Ignore config parsing errors here; converter will handle validation.
    }
  }

  const weightFile = files.find((file) => {
    const name = file.name.toLowerCase();
    return name.endsWith('.safetensors') || name.endsWith('.gguf');
  });
  if (weightFile) {
    const base = weightFile.name.replace(/\.(safetensors|gguf)$/i, '');
    if (base) return base;
  }

  return null;
}

function resolveActiveModelId() {
  const modelSelect = $('diagnostics-model');
  const selected = modelSelect?.value?.trim();
  if (selected) return selected;
  return state.activeModelId || null;
}

function normalizeModelType(value) {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  return normalized || null;
}

function isCompatibleModelType(modelType, mode) {
  const normalized = normalizeModelType(modelType);
  if (mode === 'diffusion') {
    return normalized === 'diffusion';
  }
  if (mode === 'run') {
    return normalized !== 'diffusion';
  }
  return true;
}

function isModeModelSelectable(mode) {
  return mode === 'run' || mode === 'diffusion';
}

async function getModelTypeForId(modelId) {
  if (!modelId) return null;
  const cached = state.modelTypeCache[modelId];
  if (cached) return cached;
  try {
    await openModelStore(modelId);
    const manifestText = await loadManifestFromStore();
    if (!manifestText) return null;
    const manifest = JSON.parse(manifestText);
    const modelType = normalizeModelType(manifest?.modelType) || 'transformer';
    state.modelTypeCache[modelId] = modelType;
    return modelType;
  } catch (error) {
    log.warn('DopplerDemo', `Failed to read manifest for ${modelId}: ${error.message}`);
    return null;
  }
}

async function resolveCompatibleModelId(mode) {
  if (!isModeModelSelectable(mode)) return null;
  let models = [];
  try {
    models = await listRegisteredModels();
  } catch (error) {
    log.warn('DopplerDemo', `Model registry unavailable: ${error.message}`);
  }
  const modelIds = models
    .map((entry) => entry.modelId || entry.id)
    .filter(Boolean);
  if (!modelIds.length) return null;

  const pipelineId = state.activePipelineModelId;
  if (pipelineId && modelIds.includes(pipelineId)) {
    const pipelineType = normalizeModelType(state.activePipeline?.manifest?.modelType)
      || await getModelTypeForId(pipelineId);
    if (isCompatibleModelType(pipelineType, mode)) {
      return pipelineId;
    }
  }

  const preferred = state.modeModelId?.[mode] || null;
  if (preferred && modelIds.includes(preferred)) {
    const preferredType = await getModelTypeForId(preferred);
    if (isCompatibleModelType(preferredType, mode)) {
      return preferred;
    }
  }

  const current = state.activeModelId;
  if (current && modelIds.includes(current)) {
    const currentType = await getModelTypeForId(current);
    if (isCompatibleModelType(currentType, mode)) {
      return current;
    }
  }

  for (const modelId of modelIds) {
    const modelType = await getModelTypeForId(modelId);
    if (isCompatibleModelType(modelType, mode)) {
      return modelId;
    }
  }
  return null;
}

async function syncModelForMode(mode) {
  if (!isModeModelSelectable(mode)) return;
  const compatibleId = await resolveCompatibleModelId(mode);
  if (!compatibleId) return;
  if (state.activeModelId !== compatibleId) {
    if (state.activePipeline && state.activePipelineModelId && state.activePipelineModelId !== compatibleId) {
      await unloadActivePipeline();
    }
    selectDiagnosticsModel(compatibleId);
  }
  state.modeModelId[mode] = compatibleId;
}

async function writeFileToDirectory(dirHandle, name, data) {
  const fileHandle = await dirHandle.getFileHandle(name, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(data);
  await writable.close();
}

async function exportActiveModel() {
  const modelId = resolveActiveModelId();
  if (!modelId) {
    updateExportStatus('Select an active model to export.', 0);
    return;
  }
  if (!('showDirectoryPicker' in window)) {
    updateExportStatus('Folder export requires the File System Access API.', 0);
    return;
  }

  updateExportStatus('Choose a destination folder...', 0);
  let rootHandle;
  try {
    rootHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
  } catch (error) {
    if (error.name === 'AbortError') {
      updateExportStatus('Export cancelled.', 0);
      return;
    }
    throw error;
  }

  const exportDir = await rootHandle.getDirectoryHandle(modelId, { create: true });
  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    updateExportStatus('Manifest not found in storage.', 0);
    return;
  }
  const previousManifest = getManifest();
  const manifest = parseManifest(manifestText);

  try {
    const payloads = [
      { name: 'manifest.json', data: manifestText },
    ];

    if (manifest.tensorsFile) {
      const tensorsText = await loadTensorsFromStore();
      if (!tensorsText) {
        updateExportStatus('Missing tensors.json for this model.', 0);
        return;
      }
      payloads.push({ name: manifest.tensorsFile, data: tensorsText });
    }

    const tokenizerText = await loadTokenizerFromStore();
    if (tokenizerText) {
      payloads.push({ name: 'tokenizer.json', data: tokenizerText });
    }

    const tokenizerModel = await loadTokenizerModelFromStore();
    if (tokenizerModel) {
      payloads.push({ name: 'tokenizer.model', data: tokenizerModel });
    }

    const shards = Array.isArray(manifest.shards) ? manifest.shards : [];
    const totalItems = payloads.length + shards.length;
    let completed = 0;
    const reportProgress = (name) => {
      completed += 1;
      const percent = totalItems > 0 ? (completed / totalItems) * 100 : 0;
      updateExportStatus(`Writing ${name}...`, percent);
    };

    for (const entry of payloads) {
      await writeFileToDirectory(exportDir, entry.name, entry.data);
      reportProgress(entry.name);
    }

    for (let i = 0; i < shards.length; i++) {
      const shard = shards[i];
      const index = Number.isInteger(shard.index) ? shard.index : i;
      const filename = shard.filename || `shard_${String(index).padStart(5, '0')}.bin`;
      const data = await loadShard(index);
      await writeFileToDirectory(exportDir, filename, data);
      reportProgress(filename);
    }

    updateExportStatus(`Export complete: ${modelId}`, 100);
  } finally {
    if (previousManifest) {
      setManifest(previousManifest);
    } else {
      clearManifest();
    }
  }
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

function renderModelList(models) {
  const list = $('model-list');
  if (!list) return;
  list.innerHTML = '';
  if (!models || models.length === 0) {
    list.textContent = 'No local models';
    list.classList.add('muted');
    return;
  }
  list.classList.remove('muted');
  for (const model of models) {
    const item = document.createElement('div');
    item.className = 'model-item';
    const header = document.createElement('div');
    header.className = 'model-section-header';
    const name = document.createElement('span');
    name.className = 'type-label';
    name.textContent = model.modelId || model.id || 'unknown';
    const meta = document.createElement('span');
    meta.className = 'type-caption muted';
    meta.textContent = model.quantization || model.hashAlgorithm || model.backend || '';
    header.appendChild(name);
    header.appendChild(meta);
    const detail = document.createElement('span');
    detail.className = 'type-caption muted';
    const sizeLabel = Number.isFinite(model.totalSize) ? formatBytes(model.totalSize) : 'Size: --';
    const backendLabel = model.backend ? ` - ${model.backend}` : '';
    detail.textContent = `${sizeLabel}${backendLabel}`;
    item.appendChild(header);
    item.appendChild(detail);
    item.addEventListener('click', () => {
      selectDiagnosticsModel(model.modelId || model.id || '');
    });
    list.appendChild(item);
  }
}

function updateSidebarLayout(models) {
  const panelGrid = $('panel-grid');
  if (!panelGrid) return;
  const hasModels = Array.isArray(models) && models.length > 0;
  panelGrid.dataset.layout = hasModels ? 'ready' : 'empty';
  if (!hasModels && state.uiMode !== 'models') {
    setUiMode('models');
  }
}

function selectDiagnosticsModel(modelId) {
  const modelSelect = $('diagnostics-model');
  if (!modelSelect) return;
  modelSelect.value = modelId;
  state.activeModelId = modelId || null;
  if (isModeModelSelectable(state.uiMode)) {
    state.modeModelId[state.uiMode] = modelId || null;
  }
}

async function updateStorageInfo() {
  const storageUsed = $('storage-used');
  if (!storageUsed) return;
  try {
    const info = await getQuotaInfo();
    state.storageUsageBytes = info.usage || 0;
    state.storageQuotaBytes = info.quota || 0;
    if (!info.quota) {
      storageUsed.textContent = 'Storage unavailable';
      return;
    }
    const used = formatBytes(info.usage || 0);
    const total = formatBytes(info.quota || 0);
    storageUsed.textContent = `${used} / ${total}`;
    state.lastStorageRefresh = Date.now();
  } catch (error) {
    storageUsed.textContent = `Storage error: ${error.message}`;
  }
}

async function scanOpfsDirectory(dirHandle) {
  let totalBytes = 0;
  let fileCount = 0;
  let shardCount = 0;
  let hasManifest = false;

  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file') {
      const file = await entry.getFile();
      totalBytes += file.size;
      fileCount += 1;
      if (entry.name === 'manifest.json') {
        hasManifest = true;
      }
      if (entry.name.startsWith('shard_') && entry.name.endsWith('.bin')) {
        shardCount += 1;
      }
    } else if (entry.kind === 'directory') {
      const child = await scanOpfsDirectory(entry);
      totalBytes += child.totalBytes;
      fileCount += child.fileCount;
      shardCount += child.shardCount;
      hasManifest = hasManifest || child.hasManifest;
    }
  }

  return {
    totalBytes,
    fileCount,
    shardCount,
    hasManifest,
  };
}

function setStorageInspectorStatus(message) {
  const status = $('storage-inspector-status');
  if (!status) return;
  status.textContent = message;
}

async function handleDeleteStorageEntry(entry) {
  if (!entry?.modelId) return;
  if (entry.modelId === state.activeModelId && state.activePipeline) {
    window.alert('Unload the active model before deleting its storage.');
    return;
  }
  const sizeLabel = Number.isFinite(entry.totalBytes) ? formatBytes(entry.totalBytes) : 'unknown size';
  const confirmed = window.confirm(`Delete ${entry.modelId} (${sizeLabel}) from OPFS?`);
  if (!confirmed) return;

  try {
    await deleteModel(entry.modelId);
  } catch (error) {
    log.warn('DopplerDemo', `Failed to delete ${entry.modelId}: ${error.message}`);
  }

  try {
    await removeRegisteredModel(entry.modelId);
  } catch (error) {
    log.warn('DopplerDemo', `Failed to remove registry entry: ${error.message}`);
  }

  await refreshModelList();
  await refreshStorageInspector();
}

async function refreshStorageInspector() {
  const listEl = $('storage-inspector-list');
  const backendEl = $('storage-inspector-backend');
  const summaryEl = $('storage-inspector-summary');
  const systemSection = $('storage-inspector-system-section');
  const systemList = $('storage-inspector-system');
  if (!listEl || !backendEl || !summaryEl || !systemSection || !systemList) return;
  if (state.storageInspectorScanning) return;

  state.storageInspectorScanning = true;
  listEl.innerHTML = '';
  systemList.innerHTML = '';
  systemSection.hidden = true;
  setStorageInspectorStatus('Scanning storage...');

  const runtime = getRuntimeConfig();
  const opfsRootDir = runtime?.loading?.opfsPath?.opfsRootDir || 'doppler-models';
  backendEl.textContent = navigator.storage?.getDirectory
    ? `OPFS: ${opfsRootDir}`
    : 'OPFS: unavailable';

  let registryIds = new Set();
  try {
    const registry = await listRegisteredModels();
    registryIds = new Set(
      registry.map((entry) => entry.modelId || entry.id).filter(Boolean)
    );
  } catch (error) {
    log.warn('DopplerDemo', `Registry unavailable: ${error.message}`);
  }

  try {
    if (!navigator.storage?.getDirectory) {
      summaryEl.textContent = '--';
      setStorageInspectorStatus('OPFS not available in this browser.');
      return;
    }

    const root = await navigator.storage.getDirectory();
    let opfsRoot = null;
    try {
      opfsRoot = await root.getDirectoryHandle(opfsRootDir);
    } catch (error) {
      summaryEl.textContent = '--';
      setStorageInspectorStatus('No OPFS storage found for Doppler.');
      return;
    }

    const entries = [];
    const systemEntries = [];
    for await (const entry of opfsRoot.values()) {
      if (entry.kind !== 'directory') continue;
      const stats = await scanOpfsDirectory(entry);
      if (entry.name === 'reports') {
        systemEntries.push({
          label: 'Reports',
          modelId: entry.name,
          totalBytes: stats.totalBytes,
          fileCount: stats.fileCount,
        });
      } else {
        entries.push({
          modelId: entry.name,
          totalBytes: stats.totalBytes,
          fileCount: stats.fileCount,
          shardCount: stats.shardCount,
          hasManifest: stats.hasManifest,
          registered: registryIds.has(entry.name),
        });
      }
    }

    entries.sort((a, b) => b.totalBytes - a.totalBytes);

    const totalBytes = entries.reduce((sum, entry) => sum + entry.totalBytes, 0);
    const systemBytes = systemEntries.reduce((sum, entry) => sum + entry.totalBytes, 0);
    const summaryParts = [];
    if (entries.length) {
      summaryParts.push(`Models: ${entries.length}`);
      summaryParts.push(`OPFS: ${formatBytes(totalBytes)}`);
    }
    if (systemEntries.length) {
      summaryParts.push(`Reports: ${formatBytes(systemBytes)}`);
    }
    summaryEl.textContent = summaryParts.length
      ? summaryParts.join(' • ')
      : 'No OPFS models found';

    if (!entries.length) {
      listEl.innerHTML = '<div class="type-caption muted">No OPFS models found.</div>';
      setStorageInspectorStatus('Ready');
      return;
    }

    for (const entry of entries) {
      const row = document.createElement('div');
      row.className = 'storage-entry';
      if (entry.modelId === state.activeModelId) {
        row.classList.add('is-active');
      }

      const main = document.createElement('div');
      main.className = 'storage-entry-main';

      const title = document.createElement('div');
      title.className = 'storage-entry-title';

      const name = document.createElement('span');
      name.className = 'type-caption';
      name.textContent = entry.modelId;
      title.appendChild(name);

      const tag = document.createElement('span');
      tag.className = `storage-tag${entry.registered ? '' : ' orphan'}`;
      tag.textContent = entry.registered ? 'registered' : 'orphan';
      title.appendChild(tag);

      if (!entry.hasManifest) {
        const missing = document.createElement('span');
        missing.className = 'storage-tag missing';
        missing.textContent = 'no manifest';
        title.appendChild(missing);
      }

      main.appendChild(title);

      const shardLabel = entry.shardCount
        ? `${entry.shardCount} shards`
        : `${entry.fileCount} files`;
      const detail = document.createElement('span');
      detail.className = 'type-caption muted';
      detail.textContent = `${formatBytes(entry.totalBytes)} • ${shardLabel}`;
      main.appendChild(detail);

      const actions = document.createElement('div');
      actions.className = 'storage-entry-actions';

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'btn btn-small';
      deleteBtn.type = 'button';
      deleteBtn.textContent = 'Delete';
      if (entry.modelId === state.activeModelId && state.activePipeline) {
        deleteBtn.disabled = true;
        deleteBtn.title = 'Unload the active model before deleting.';
      }
      deleteBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        await handleDeleteStorageEntry(entry);
      });
      actions.appendChild(deleteBtn);

      row.appendChild(main);
      row.appendChild(actions);
      row.addEventListener('click', () => selectDiagnosticsModel(entry.modelId));
      listEl.appendChild(row);
    }

    if (systemEntries.length) {
      systemSection.hidden = false;
      for (const entry of systemEntries) {
        const row = document.createElement('div');
        row.className = 'storage-entry';

        const main = document.createElement('div');
        main.className = 'storage-entry-main';

        const title = document.createElement('div');
        title.className = 'storage-entry-title';

        const name = document.createElement('span');
        name.className = 'type-caption';
        name.textContent = entry.label;
        title.appendChild(name);

        const tag = document.createElement('span');
        tag.className = 'storage-tag system';
        tag.textContent = 'system';
        title.appendChild(tag);

        main.appendChild(title);

        const detail = document.createElement('span');
        detail.className = 'type-caption muted';
        detail.textContent = `${formatBytes(entry.totalBytes)} • ${entry.fileCount} files`;
        main.appendChild(detail);

        row.appendChild(main);
        systemList.appendChild(row);
      }
    }

    setStorageInspectorStatus('Ready');
  } catch (error) {
    summaryEl.textContent = '--';
    setStorageInspectorStatus(`Storage scan failed: ${error.message}`);
  } finally {
    state.storageInspectorScanning = false;
    state.storageInspectorLastScan = Date.now();
  }
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
  } else {
    for (const model of models) {
      const opt = document.createElement('option');
      opt.value = model.modelId || model.id || '';
      opt.textContent = model.modelId || model.id || 'unknown';
      modelSelect.appendChild(opt);
    }
  }
  renderModelList(models);
  updateSidebarLayout(models);
  await updateStorageInfo();
  await syncModelForMode(state.uiMode);
  if (state.uiMode === 'models') {
    await refreshStorageInspector();
  }
}

async function refreshGpuInfo() {
  const deviceEl = $('gpu-device');
  const ramRow = $('gpu-ram-row');
  const ramEl = $('gpu-ram');
  const vramEl = $('gpu-vram');
  const featuresEl = $('gpu-features');
  const vramLabel = $('gpu-vram-label');
  const unifiedNote = $('gpu-unified-note');

  if (!isWebGPUAvailable()) {
    setText(deviceEl, 'WebGPU unavailable');
    setText(vramEl, '--');
    setText(featuresEl, 'none');
    setHidden(ramRow, true);
    setHidden(unifiedNote, true);
    return;
  }

  try {
    await initDevice();
    const caps = getKernelCapabilities();
    const adapter = caps.adapterInfo || {};
    const deviceLabel = [adapter.vendor, adapter.architecture || adapter.device, adapter.description]
      .filter(Boolean)
      .join(' ');
    setText(deviceEl, deviceLabel || 'Unknown GPU');

    if (Number.isFinite(navigator.deviceMemory)) {
      state.systemMemoryBytes = navigator.deviceMemory * 1024 * 1024 * 1024;
      setText(ramEl, `${navigator.deviceMemory} GB`);
      setHidden(ramRow, false);
    } else {
      setHidden(ramRow, true);
    }

    state.gpuMaxBytes = caps.maxBufferSize || 0;
    if (vramLabel) vramLabel.textContent = 'Buffer Limit';
    setText(vramEl, caps.maxBufferSize ? formatBytes(caps.maxBufferSize) : '--');

    const features = [
      caps.hasF16 && 'f16',
      caps.hasSubgroups && 'subgroups',
      caps.hasSubgroupsF16 && 'subgroups-f16',
      caps.hasTimestampQuery && 'timestamp',
    ].filter(Boolean);
    setText(featuresEl, features.length ? features.join(', ') : 'basic');

    let preferUnified = false;
    try {
      const platformConfig = getPlatformConfig();
      preferUnified = Boolean(platformConfig?.platform?.memoryHints?.preferUnifiedMemory);
    } catch {
      preferUnified = false;
    }
    setHidden(unifiedNote, !preferUnified);
  } catch (error) {
    setText(deviceEl, `GPU init failed`);
    setText(vramEl, '--');
    setText(featuresEl, 'none');
    setHidden(ramRow, true);
    setHidden(unifiedNote, true);
    log.warn('DopplerDemo', `GPU init failed: ${error.message}`);
  }
}

function updatePerformancePanel(snapshot) {
  const tpsEl = $('stat-tps');
  const memoryEl = $('stat-memory');
  const gpuEl = $('stat-gpu');
  const kvEl = $('stat-kv');
  const e2eEl = $('stat-e2e');
  const prefillEl = $('stat-prefill');
  const decodeEl = $('stat-decode');

  const metrics = state.lastMetrics || {};
  const liveTps = state.chatGenerating ? metrics.liveTokensPerSec : null;
  const tps = Number.isFinite(liveTps)
    ? liveTps
    : (Number.isFinite(metrics.tokensPerSec)
      ? metrics.tokensPerSec
      : (Number.isFinite(metrics.medianTokensPerSec) ? metrics.medianTokensPerSec : null));
  setText(tpsEl, tps !== null ? `${tps.toFixed(2)}` : '--');

  const gpuBytes = snapshot?.gpu?.currentBytes ?? null;
  const usedBytes = Number.isFinite(state.lastMemoryStats?.used)
    ? state.lastMemoryStats.used
    : gpuBytes;
  setText(memoryEl, Number.isFinite(usedBytes) ? formatBytes(usedBytes) : '--');

  const activeBuffers = snapshot?.gpu?.activeBuffers ?? null;
  const pooledBuffers = snapshot?.gpu?.pooledBuffers ?? null;
  if (Number.isFinite(activeBuffers) && Number.isFinite(pooledBuffers)) {
    setText(gpuEl, `${activeBuffers}/${pooledBuffers}`);
  } else {
    setText(gpuEl, '--');
  }

  const kvStats = state.lastMemoryStats?.kvCache ?? null;
  const kvBytes = kvStats?.allocated ?? null;
  const kvSeq = kvStats?.seqLen ?? null;
  const kvMax = kvStats?.maxSeqLen ?? null;
  if (Number.isFinite(kvBytes)) {
    const usage = Number.isFinite(kvSeq) && Number.isFinite(kvMax) ? ` (${kvSeq}/${kvMax})` : '';
    setText(kvEl, `${formatBytes(kvBytes)}${usage}`);
  } else {
    setText(kvEl, '--');
  }

  const stats = state.lastInferenceStats || {};
  const prefillTokens = Number.isFinite(stats.prefillTokens) ? stats.prefillTokens : null;
  const prefillTime = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null;
  const ttftMs = Number.isFinite(stats.ttftMs) ? stats.ttftMs : prefillTime;
  const prefillRate = (prefillTokens != null && prefillTime && prefillTime > 0)
    ? prefillTokens / (prefillTime / 1000)
    : null;
  const decodeTokens = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
  const decodeTime = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
  const e2eTime = (Number.isFinite(stats.totalTimeMs) && stats.totalTimeMs > 0)
    ? stats.totalTimeMs
    : (Number.isFinite(prefillTime) && Number.isFinite(decodeTime) ? prefillTime + decodeTime : null);
  const e2eRate = (decodeTokens != null && e2eTime && e2eTime > 0)
    ? decodeTokens / (e2eTime / 1000)
    : null;
  if (e2eEl) {
    setText(e2eEl, formatRate(e2eRate));
  }
  if (prefillEl) {
    if (prefillTokens == null && ttftMs == null && prefillRate == null) {
      setText(prefillEl, '--');
    } else {
      const ttftLabel = ttftMs != null ? `TTFT ${Math.round(ttftMs)}ms` : 'TTFT --';
      const tokenLabel = prefillTokens != null ? `${prefillTokens} tok` : '--';
      const rateLabel = prefillRate != null ? `${prefillRate.toFixed(2)} tok/s` : '--';
      setText(prefillEl, `${ttftLabel} - ${tokenLabel} @ ${rateLabel}`);
    }
  }

  if (decodeEl) {
    if (decodeTokens == null && decodeTime == null) {
      setText(decodeEl, '--');
    } else {
      const tokenLabel = decodeTokens != null ? `${decodeTokens} tok` : '--';
      const rateLabel = (decodeTokens != null && decodeTime && decodeTime > 0)
        ? `${(decodeTokens / (decodeTime / 1000)).toFixed(2)} tok/s`
        : '--';
      setText(decodeEl, `${tokenLabel} - ${rateLabel}`);
    }
  }
}

function updateMemoryPanel(snapshot) {
  const hasHeap = Number.isFinite(snapshot?.jsHeapUsed);
  const jsHeapUsed = hasHeap ? snapshot.jsHeapUsed : 0;
  const jsHeapLimit = Number.isFinite(snapshot?.jsHeapLimit) ? snapshot.jsHeapLimit : 0;
  const hasGpu = Number.isFinite(snapshot?.gpu?.currentBytes);
  const gpuBytes = hasGpu ? snapshot.gpu.currentBytes : 0;
  const gpuLimit = state.gpuMaxBytes || 0;
  const kvStats = state.lastMemoryStats?.kvCache || null;
  const hasKv = Number.isFinite(kvStats?.allocated);
  const kvBytes = hasKv ? kvStats.allocated : 0;

  const totalCapacity = state.systemMemoryBytes || (jsHeapLimit + gpuLimit) || gpuLimit || jsHeapLimit || 0;
  const totalUsed = jsHeapUsed + gpuBytes;
  const headroom = totalCapacity > 0 ? Math.max(0, totalCapacity - totalUsed) : 0;

  setText($('memory-total'), totalCapacity ? `${formatBytes(totalUsed)} / ${formatBytes(totalCapacity)}` : '--');
  setBarWidth('memory-bar-heap-stacked', totalCapacity ? (jsHeapUsed / totalCapacity) * 100 : 0);
  setBarWidth('memory-bar-gpu-stacked', totalCapacity ? (gpuBytes / totalCapacity) * 100 : 0);

  if (hasHeap) {
    setText($('memory-heap'), jsHeapLimit ? `${formatBytes(jsHeapUsed)} / ${formatBytes(jsHeapLimit)}` : formatBytes(jsHeapUsed));
  } else {
    setText($('memory-heap'), '--');
  }
  setBarWidth('memory-bar-heap', jsHeapLimit ? (jsHeapUsed / jsHeapLimit) * 100 : (totalCapacity ? (jsHeapUsed / totalCapacity) * 100 : 0));

  if (hasGpu) {
    setText($('memory-gpu'), gpuLimit ? `${formatBytes(gpuBytes)} / ${formatBytes(gpuLimit)}` : formatBytes(gpuBytes));
  } else {
    setText($('memory-gpu'), '--');
  }
  setBarWidth('memory-bar-gpu', gpuLimit ? (gpuBytes / gpuLimit) * 100 : (totalCapacity ? (gpuBytes / totalCapacity) * 100 : 0));

  const kvLabel = hasKv ? formatBytes(kvBytes) : '--';
  const kvMeta = kvStats?.seqLen && kvStats?.maxSeqLen ? ` (${kvStats.seqLen}/${kvStats.maxSeqLen})` : '';
  setText($('memory-kv'), kvLabel + kvMeta);
  setBarWidth('memory-bar-kv', gpuLimit ? (kvBytes / gpuLimit) * 100 : 0);

  const storageUsage = state.storageUsageBytes || 0;
  const storageQuota = state.storageQuotaBytes || 0;
  setText($('memory-opfs'), storageQuota ? `${formatBytes(storageUsage)} / ${formatBytes(storageQuota)}` : '--');
  setBarWidth('memory-bar-opfs', storageQuota ? (storageUsage / storageQuota) * 100 : 0);

  setText($('memory-headroom'), totalCapacity ? formatBytes(headroom) : '--');
  setBarWidth('memory-bar-headroom', totalCapacity ? (headroom / totalCapacity) * 100 : 0);
}

function updateMemoryControls() {
  const unloadBtn = $('unload-model-btn');
  if (unloadBtn) {
    unloadBtn.disabled = !state.activePipeline;
  }
}

function getSelectedModelId() {
  if (state.activeModelId) return state.activeModelId;
  const modelSelect = $('diagnostics-model');
  const selected = modelSelect?.value || '';
  if (selected) {
    state.activeModelId = selected;
    return selected;
  }
  if (modelSelect?.options?.length) {
    const fallback = modelSelect.options[0].value;
    state.activeModelId = fallback || null;
    return fallback || null;
  }
  return null;
}

function setChatGenerating(isGenerating) {
  state.chatGenerating = Boolean(isGenerating);
  const chatInput = $('chat-input');
  const sendBtn = $('send-btn');
  const stopBtn = $('stop-btn');
  if (chatInput) chatInput.disabled = state.chatGenerating || state.chatLoading;
  if (sendBtn) sendBtn.disabled = state.chatGenerating || state.chatLoading;
  if (stopBtn) setHidden(stopBtn, !state.chatGenerating);
  updateStatusIndicator();
}

function setChatLoading(isLoading) {
  state.chatLoading = Boolean(isLoading);
  const chatInput = $('chat-input');
  const sendBtn = $('send-btn');
  if (chatInput) chatInput.disabled = state.chatGenerating || state.chatLoading;
  if (sendBtn) sendBtn.disabled = state.chatGenerating || state.chatLoading;
  updateStatusIndicator();
}

function scrollChatToBottom() {
  const container = $('chat-messages');
  if (!container) return;
  container.scrollTop = container.scrollHeight;
}

function appendChatMessage(role, content) {
  const container = $('chat-messages');
  if (!container) return null;
  const message = document.createElement('div');
  message.className = `message message-${role}`;
  const label = document.createElement('div');
  label.className = 'message-role type-label';
  label.textContent = role === 'assistant' ? 'Assistant' : 'User';
  const body = document.createElement('div');
  body.className = 'message-content';
  body.textContent = content || '';
  message.appendChild(label);
  message.appendChild(body);
  container.appendChild(message);
  scrollChatToBottom();
  return body;
}

function clearChatMessages() {
  const container = $('chat-messages');
  if (container) {
    container.innerHTML = '';
  }
  state.chatMessages = [];
}

function renderRunLog() {
  const container = $('run-log-rows');
  if (!container) return;
  container.innerHTML = '';
  for (const entry of state.runLog) {
    const row = document.createElement('div');
    row.className = 'run-log-row';
    const cells = [
      entry.label,
      formatMs(entry.ttftMs),
      formatRate(entry.prefillRate),
      formatRate(entry.decodeRate),
      formatRate(entry.e2eRate),
    ];
    for (const value of cells) {
      const cell = document.createElement('span');
      cell.textContent = value;
      row.appendChild(cell);
    }
    container.appendChild(row);
  }
}

function recordRunLog(stats, label) {
  if (!stats) return;
  const prefillTokens = Number.isFinite(stats.prefillTokens) ? stats.prefillTokens : null;
  const decodeTokens = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
  const prefillTime = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null;
  const decodeTime = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
  const totalTime = Number.isFinite(stats.totalTimeMs)
    ? stats.totalTimeMs
    : ((prefillTime && decodeTime) ? prefillTime + decodeTime : null);
  const entry = {
    label,
    ttftMs: Number.isFinite(stats.ttftMs) ? stats.ttftMs : prefillTime,
    prefillRate: (prefillTokens != null && prefillTime && prefillTime > 0)
      ? prefillTokens / (prefillTime / 1000)
      : null,
    decodeRate: (decodeTokens != null && decodeTime && decodeTime > 0)
      ? decodeTokens / (decodeTime / 1000)
      : null,
    e2eRate: (decodeTokens != null && totalTime && totalTime > 0)
      ? decodeTokens / (totalTime / 1000)
      : null,
  };
  state.runLog.unshift(entry);
  state.runLog = state.runLog.slice(0, 8);
  renderRunLog();
}

function buildSamplingOverride() {
  const tempInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const temperature = Number.parseFloat(tempInput?.value ?? '');
  const topP = Number.parseFloat(topPInput?.value ?? '');
  const topK = Number.parseInt(topKInput?.value ?? '', 10);
  const sampling = {};
  let hasValue = false;
  if (Number.isFinite(temperature)) {
    sampling.temperature = Math.max(0, temperature);
    hasValue = true;
  }
  if (Number.isFinite(topP)) {
    sampling.topP = Math.max(0, Math.min(1, topP));
    hasValue = true;
  }
  if (Number.isFinite(topK)) {
    sampling.topK = Math.max(0, topK);
    hasValue = true;
  }
  if (!hasValue) return null;
  return {
    inference: {
      sampling,
    },
  };
}

function syncSamplingInputsFromRuntime() {
  if (state.runtimeSamplingOverride) return;
  const tempInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  if (!tempInput || !topPInput || !topKInput) return;
  const sampling = getRuntimeConfig().inference?.sampling || {};
  state.samplingSyncing = true;
  tempInput.value = Number.isFinite(sampling.temperature) ? String(sampling.temperature) : '';
  topPInput.value = Number.isFinite(sampling.topP) ? String(sampling.topP) : '';
  topKInput.value = Number.isFinite(sampling.topK) ? String(sampling.topK) : '';
  state.samplingSyncing = false;
}

function applySamplingOverrideFromInputs() {
  if (state.samplingSyncing) return;
  const override = buildSamplingOverride();
  state.runtimeSamplingOverride = override;
  state.runtimeSamplingLabel = override ? 'sampling' : null;
  applySelectedRuntimePreset();
}

function showProgressOverlay(title) {
  const overlay = $('progress-overlay');
  const titleEl = $('progress-title');
  if (titleEl && title) titleEl.textContent = title;
  setProgressPhase('source', 0, '--');
  setProgressPhase('gpu', 0, '--');
  if (overlay) overlay.hidden = false;
}

function hideProgressOverlay() {
  const overlay = $('progress-overlay');
  if (overlay) overlay.hidden = true;
}

function setProgressPhase(phase, percent, label) {
  const row = document.querySelector(`.progress-phase-row[data-phase="${phase}"]`);
  if (!row) return;
  const fill = row.querySelector('.progress-fill');
  const value = row.querySelector('.progress-phase-value');
  if (fill) fill.style.width = `${clampPercent(percent)}%`;
  if (value) value.textContent = label ?? `${Math.round(clampPercent(percent))}%`;
}

function updateProgressFromLoader(info) {
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

async function loadPipelineFromStorage(modelId) {
  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage');
  }
  const manifest = parseManifest(manifestText);
  await initDevice();
  const device = getDevice();
  return createPipeline(manifest, {
    gpu: { device },
    storage: { loadShard },
    runtimeConfig: getRuntimeConfig(),
    onProgress: (progress) => updateProgressFromLoader(progress),
  });
}

async function ensureChatPipeline() {
  const modelId = getSelectedModelId();
  if (!modelId) {
    throw new Error('Select a model before chatting');
  }
  if (state.activePipeline && state.activeModelId === modelId) {
    return state.activePipeline;
  }
  if (state.activePipeline) {
    await unloadActivePipeline();
  }
  showProgressOverlay('Loading Model');
  setChatLoading(true);
  try {
    const pipeline = await loadPipelineFromStorage(modelId);
    state.activePipeline = pipeline;
    state.activeModelId = modelId;
    state.activePipelineModelId = modelId;
    if (pipeline?.manifest?.modelType) {
      state.modelTypeCache[modelId] = normalizeModelType(pipeline.manifest.modelType);
    }
    state.modeModelId.run = modelId;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? null;
    updateMemoryControls();
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    return pipeline;
  } finally {
    hideProgressOverlay();
    setChatLoading(false);
  }
}

async function ensureDiffusionPipeline() {
  const modelId = getSelectedModelId();
  if (!modelId) {
    throw new Error('Select a model before generating');
  }
  if (state.activePipeline && state.activeModelId === modelId) {
    return state.activePipeline;
  }
  if (state.activePipeline) {
    await unloadActivePipeline();
  }
  showProgressOverlay('Loading Diffusion Model');
  state.diffusionLoading = true;
  updateStatusIndicator();
  try {
    const pipeline = await loadPipelineFromStorage(modelId);
    state.activePipeline = pipeline;
    state.activeModelId = modelId;
    state.activePipelineModelId = modelId;
    if (pipeline?.manifest?.modelType) {
      state.modelTypeCache[modelId] = normalizeModelType(pipeline.manifest.modelType);
    }
    state.modeModelId.diffusion = modelId;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? null;
    updateMemoryControls();
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    return pipeline;
  } finally {
    hideProgressOverlay();
    state.diffusionLoading = false;
    updateStatusIndicator();
  }
}

function drawDiffusionCanvas(result) {
  const canvas = $('diffusion-canvas');
  if (!canvas || !result) return;
  canvas.width = result.width;
  canvas.height = result.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const imageData = new ImageData(result.pixels, result.width, result.height);
  ctx.putImageData(imageData, 0, 0);
}

async function handleDiffusionRun() {
  if (state.diffusionGenerating || state.diffusionLoading) return;
  const promptEl = $('diffusion-prompt');
  const negativeEl = $('diffusion-negative');
  const stepsEl = $('diffusion-steps');
  const guidanceEl = $('diffusion-guidance');
  const seedEl = $('diffusion-seed');
  const widthEl = $('diffusion-width');
  const heightEl = $('diffusion-height');

  const request = {
    prompt: promptEl?.value?.trim() || '',
    negativePrompt: negativeEl?.value?.trim() || '',
    steps: stepsEl?.value ? Number(stepsEl.value) : undefined,
    guidanceScale: guidanceEl?.value ? Number(guidanceEl.value) : undefined,
    seed: seedEl?.value ? Number(seedEl.value) : undefined,
    width: widthEl?.value ? Number(widthEl.value) : undefined,
    height: heightEl?.value ? Number(heightEl.value) : undefined,
  };

  updateDiffusionStatus('Preparing...');
  state.diffusionGenerating = true;
  updateStatusIndicator();
  try {
    const pipeline = await ensureDiffusionPipeline();
    if (!pipeline.generate) {
      throw new Error('Selected model does not support diffusion generation.');
    }
    if (!pipeline.manifest || pipeline.manifest.modelType !== 'diffusion') {
      throw new Error('Selected model is not a diffusion model.');
    }
    updateDiffusionStatus('Generating...');
    const result = await pipeline.generate(request);
    if (!Number.isFinite(result?.width) || result.width <= 0 || !Number.isFinite(result?.height) || result.height <= 0) {
      throw new Error('Diffusion output dimensions are invalid.');
    }
    drawDiffusionCanvas(result);
    state.lastInferenceStats = pipeline.getStats?.() ?? null;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? state.lastMemoryStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    updateDiffusionStatus('Complete');
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
  } catch (error) {
    log.error('DopplerDemo', `Diffusion run failed: ${error.message}`);
    updateDiffusionStatus(`Error: ${error.message}`);
  } finally {
    state.diffusionGenerating = false;
    updateStatusIndicator();
  }
}

function handleDiffusionClear() {
  const canvas = $('diffusion-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }
  updateDiffusionStatus('Idle');
}

async function handleChatSend() {
  if (state.chatGenerating || state.chatLoading) return;
  const input = $('chat-input');
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;

  appendChatMessage('user', text);
  state.chatMessages.push({ role: 'user', content: text });
  input.value = '';

  const assistantEl = appendChatMessage('assistant', '');
  const messageIndex = state.chatMessages.length;
  state.chatMessages.push({ role: 'assistant', content: '' });

  let pipeline;
  try {
    pipeline = await ensureChatPipeline();
  } catch (error) {
    if (assistantEl) assistantEl.textContent = error.message;
    if (state.chatMessages[messageIndex]) {
      state.chatMessages[messageIndex].content = error.message;
    }
    return;
  }

  const controller = new AbortController();
  state.chatAbortController = controller;
  setChatGenerating(true);

  const templateType = pipeline?.modelConfig?.chatTemplateType ?? null;
  const prompt = formatChatMessages(state.chatMessages, templateType);
  let output = '';
  let tokenCount = 0;
  const start = performance.now();
  let firstTokenAt = null;

  try {
    for await (const token of pipeline.generate(prompt, {
      useChatTemplate: false,
      signal: controller.signal,
    })) {
      if (controller.signal.aborted) break;
      output += token;
      tokenCount += 1;
      const now = performance.now();
      if (!firstTokenAt) {
        firstTokenAt = now;
      }
      if (firstTokenAt) {
        const elapsedDecode = Math.max(1, now - firstTokenAt);
        const liveTokensPerSec = tokenCount / (elapsedDecode / 1000);
        state.lastMetrics = {
          ...(state.lastMetrics || {}),
          liveTokensPerSec,
        };
      }
      if (assistantEl) assistantEl.textContent = output;
      scrollChatToBottom();
    }
  } catch (error) {
    if (assistantEl) assistantEl.textContent = `Error: ${error.message}`;
  } finally {
    const elapsed = Math.max(1, performance.now() - start);
    const tokensPerSec = tokenCount > 0 ? Number(((tokenCount / elapsed) * 1000).toFixed(2)) : null;
    state.lastMetrics = {
      ...(state.lastMetrics || {}),
      tokensPerSec,
      liveTokensPerSec: null,
    };
    if (state.chatMessages[messageIndex]) {
      state.chatMessages[messageIndex].content = output;
    }
    state.lastMemoryStats = pipeline?.getMemoryStats?.() ?? state.lastMemoryStats;
    state.lastInferenceStats = pipeline?.getStats?.() ?? state.lastInferenceStats;
    state.runCounter += 1;
    recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    setChatGenerating(false);
    state.chatAbortController = null;
  }
}

function stopChatGeneration() {
  if (state.chatAbortController) {
    state.chatAbortController.abort();
  }
}

async function unloadActivePipeline() {
  if (!state.activePipeline) return;
  try {
    await state.activePipeline.unload?.();
  } catch (error) {
    log.warn('DopplerDemo', `Unload failed: ${error.message}`);
  }
  state.activePipeline = null;
  state.activePipelineModelId = null;
  state.lastMemoryStats = null;
  state.lastInferenceStats = null;
  updateMemoryControls();
  const snapshot = captureMemorySnapshot();
  updateMemoryPanel(snapshot);
  updatePerformancePanel(snapshot);
}

async function clearAllMemory() {
  await unloadActivePipeline();
  destroyBufferPool();
  const snapshot = captureMemorySnapshot();
  updateMemoryPanel(snapshot);
  updatePerformancePanel(snapshot);
}

function startTelemetryLoop() {
  if (state.uiIntervalId) return;
  state.uiIntervalId = setInterval(async () => {
    const now = Date.now();
    if (now - state.lastStorageRefresh > 15000) {
      state.lastStorageRefresh = now;
      await updateStorageInfo();
    }
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
  }, 1000);
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

function populateRuntimeConfigPresets() {
  const presetSelect = $('runtime-config-preset');
  if (!presetSelect) return;
  presetSelect.innerHTML = '';
  for (const entry of RUNTIME_CONFIG_PRESETS) {
    const opt = document.createElement('option');
    opt.value = entry.id;
    opt.textContent = entry.label;
    presetSelect.appendChild(opt);
  }
}

function buildConverterConfig() {
  const presetSelect = $('convert-model-preset');
  const presetId = presetSelect?.value?.trim() || null;
  const weightSelect = $('convert-weight-dtype');
  const weightOverride = weightSelect?.value?.trim().toLowerCase() || null;

  const config = createConverterConfig();
  if (presetId) {
    config.presets.model = presetId;
  }
  if (weightOverride) {
    config.quantization.weights = weightOverride;
  }
  return config;
}

async function runConversion(files, converterConfig, label, modelIdOverride) {
  if (!isConversionSupported()) {
    throw new Error('Browser conversion requires OPFS or IndexedDB.');
  }
  updateConvertStatus(`Preparing conversion${label ? ` (${label})` : ''}...`, 0);
  state.convertActive = true;
  updateStatusIndicator();
  try {
    const resultModelId = await convertModel(files, {
      modelId: modelIdOverride || undefined,
      converterConfig,
      onProgress: (update) => {
        if (!update) return;
        const percent = Number.isFinite(update.percent) ? update.percent : null;
        const message = update.message || 'Converting...';
        updateConvertStatus(label ? `${message} (${label})` : message, percent);
      },
    });
    updateConvertStatus(`Conversion complete: ${resultModelId}`, 100);
    await refreshModelList();
  } finally {
    state.convertActive = false;
    updateStatusIndicator();
  }
}

async function regenerateManifest(modelId) {
  if (!modelId) {
    throw new Error('Select a model before regenerating the manifest.');
  }

  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage.');
  }

  const manifest = parseManifest(manifestText);
  let tensorMap = manifest.tensors ?? null;
  if (!tensorMap && manifest.tensorsFile) {
    const tensorsText = await loadTensorsFromStore();
    if (!tensorsText) {
      throw new Error('tensors.json not found in storage.');
    }
    tensorMap = JSON.parse(tensorsText);
  }
  if (!tensorMap) {
    throw new Error('Manifest is missing tensor locations.');
  }

  const tensorNames = Object.keys(tensorMap);
  for (const name of tensorNames) {
    const entry = tensorMap[name];
    if (entry) {
      entry.role = classifyTensorRole(name);
    }
  }

  let inference = manifest.inference;
  if (manifest.modelType === 'diffusion') {
    if (!inference) {
      inference = { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'diffusion' };
    }
  } else {
    const rawConfig = manifest.config ?? {};
    const architectureHint = rawConfig.architectures?.[0] ?? rawConfig.model_type ?? '';
    const presetId = manifest.inference?.presetId || detectPreset(rawConfig, architectureHint);
    if (presetId === 'transformer') {
      const modelType = rawConfig.model_type ?? 'unknown';
      throw new Error(
        `Unknown model family: architecture="${architectureHint || 'unknown'}", model_type="${modelType}"`
      );
    }
    const preset = resolvePreset(presetId);
    const headDim = rawConfig.head_dim
      ?? (manifest.architecture && typeof manifest.architecture === 'object' ? manifest.architecture.headDim : null);
    if (!headDim) {
      throw new Error('Missing headDim in manifest config.');
    }
    inference = buildManifestInference(
      preset,
      rawConfig,
      headDim,
      manifest.quantizationInfo ?? null,
      tensorNames
    );
  }

  const embeddingOutput = inferEmbeddingOutputConfig(tensorMap);
  if (embeddingOutput && inference?.output) {
    inference = {
      ...inference,
      output: {
        ...inference.output,
        ...embeddingOutput,
      },
    };
  }

  const updatedManifest = {
    ...manifest,
    inference,
    tensors: tensorMap,
    tensorCount: tensorNames.length,
    metadata: {
      ...(manifest.metadata || {}),
      manifestRegeneratedAt: new Date().toISOString(),
    },
  };

  await saveManifest(JSON.stringify(updatedManifest, null, 2));
  if (manifest.tensorsFile) {
    await saveTensorsToStore(JSON.stringify(tensorMap, null, 2));
  }

  return updatedManifest;
}

async function handleRegenerateManifest() {
  if (state.convertActive) return;
  const modelId = getSelectedModelId();
  updateConvertStatus(`Regenerating manifest${modelId ? ` (${modelId})` : ''}...`, 0);
  state.convertActive = true;
  updateStatusIndicator();
  try {
    await regenerateManifest(modelId);
    updateConvertStatus(`Manifest regenerated: ${modelId}`, 100);
    await refreshModelList();
  } catch (error) {
    log.error('DopplerDemo', `Manifest regenerate failed: ${error.message}`);
    updateConvertStatus(`Manifest error: ${error.message}`, 0);
  } finally {
    state.convertActive = false;
    updateStatusIndicator();
  }
}

async function handleConvertFiles() {
  updateConvertStatus('Select a model folder or files...', 0);
  let files = null;
  let pickedLabel = null;
  try {
    const pickedDirectory = await pickModelDirectory();
    files = pickedDirectory?.files || null;
    pickedLabel = pickedDirectory?.directoryName || null;
  } catch (error) {
    files = null;
  }

  if (!files || files.length === 0) {
    const pickedFiles = await pickModelFiles({ multiple: true });
    files = pickedFiles?.files || null;
  }

  if (!files || files.length === 0) {
    updateConvertStatus('No model files found in the selected folder.', 0);
    return;
  }

  const hasWeights = files.some((file) => {
    const name = file.name.toLowerCase();
    return name.endsWith('.safetensors') || name.endsWith('.gguf');
  });
  if (!hasWeights) {
    updateConvertStatus('Missing .safetensors or .gguf in the selected folder.', 0);
    return;
  }

  const modelIdOverride = await deriveModelIdFromFiles(files, pickedLabel);
  if (!modelIdOverride) {
    updateConvertStatus('Missing modelId. Rename the folder or provide config.json.', 0);
    return;
  }

  updateConvertStatus(
    `Found ${files.length} files${pickedLabel ? ` in ${pickedLabel}` : ''}. Starting conversion...`,
    0
  );
  const converterConfig = buildConverterConfig();
  await runConversion(files, converterConfig, pickedLabel, modelIdOverride);
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

function updateDownloadStatus(progress) {
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
  const isComplete = statusText.toLowerCase().includes('complete');
  state.downloadActive = !isComplete;
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
    stats.textContent = `${percent.toFixed(1)}% - ${entry.status || 'idle'}`;
    row.appendChild(name);
    row.appendChild(stats);
    container.appendChild(row);
  }
}

async function refreshDownloads() {
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

function updateRuntimeConfigStatus(presetId) {
  const status = $('runtime-config-status');
  if (!status) return;
  const presetLabel = presetId || 'default';
  if (state.runtimeOverride) {
    const labels = [];
    if (state.runtimeOverrideLabel) {
      labels.push(state.runtimeOverrideLabel);
    }
    if (state.runtimeSamplingLabel) {
      labels.push(state.runtimeSamplingLabel);
    }
    const overrideLabel = labels.length ? labels.join(' + ') : 'custom';
    status.textContent = `Preset: ${presetLabel} - Override: ${overrideLabel}`;
    return;
  }
  status.textContent = `Preset: ${presetLabel}`;
}

async function setRuntimeOverride(runtime, label) {
  state.runtimeOverrideBase = runtime;
  state.runtimeOverrideLabel = label || null;
  await applySelectedRuntimePreset();
}

async function handleRuntimeConfigFile(file) {
  if (!file) return;
  try {
    const text = await file.text();
    const json = JSON.parse(text);
    const runtime = normalizeRuntimeConfig(json);
    if (!runtime) {
      throw new Error('Runtime config file is missing runtime fields');
    }
    await setRuntimeOverride(runtime, file.name);
    const presetSelect = $('runtime-config-preset');
    if (presetSelect) {
      presetSelect.value = '';
    }
  } catch (error) {
    updateDiagnosticsStatus(`Runtime config error: ${error.message}`, true);
  }
}

async function applyRuntimeConfigPreset(presetId) {
  if (!presetId) {
    state.runtimeOverrideBase = null;
    state.runtimeOverrideLabel = null;
    await applySelectedRuntimePreset();
    return;
  }
  try {
    const { runtime } = await loadRuntimePreset(presetId);
    await setRuntimeOverride(runtime, presetId);
  } catch (error) {
    updateDiagnosticsStatus(`Runtime config preset error: ${error.message}`, true);
  }
}

function getMergedRuntimeOverride() {
  return mergeRuntimeOverrides(state.runtimeOverrideBase, state.runtimeSamplingOverride);
}

async function applySelectedRuntimePreset() {
  const presetSelect = $('runtime-preset');
  if (!presetSelect) return;
  const presetId = presetSelect.value || 'default';
  try {
    await applyRuntimePreset(presetId);
    const mergedOverride = getMergedRuntimeOverride();
    state.runtimeOverride = mergedOverride;
    if (mergedOverride) {
      const mergedRuntime = mergeRuntimeOverrides(getRuntimeConfig(), mergedOverride);
      setRuntimeConfig(mergedRuntime);
    }
    updateRuntimeConfigStatus(presetId);
    syncSamplingInputsFromRuntime();
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
        runtimeConfig: getRuntimeConfig(),
      });
      updateDiagnosticsStatus('Verified');
      return;
    }

    if (state.activePipeline) {
      await unloadActivePipeline();
    }

    const options = {
      suite,
      runtimePreset,
      modelId,
    };
    if (state.runtimeOverride) {
      options.runtimeConfig = getRuntimeConfig();
    }
    const result = await controller.runSuite(
      modelId ? { sources: { browser: { id: modelId } } } : null,
      { ...options, keepPipeline: true }
    );
    state.lastReport = result.report;
    state.lastReportInfo = result.reportInfo;
    state.lastMetrics = result.metrics ?? null;
    if (result.memoryStats) {
      state.lastMemoryStats = result.memoryStats;
    }
    if (result.pipeline !== undefined) {
      state.activePipeline = result.pipeline;
    }
    state.activeModelId = modelId || null;
    state.lastInferenceStats = result.pipeline?.getStats?.() ?? state.lastInferenceStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    updateDiagnosticsStatus(`Complete (${result.suite})`);
    if (result.reportInfo?.path) {
      updateDiagnosticsReport(result.reportInfo.path);
    } else if (result.report?.timestamp) {
      updateDiagnosticsReport(result.report.timestamp);
    }
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    updateMemoryControls();
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
  const regenManifestBtn = $('regen-manifest-btn');
  const downloadStart = $('download-start-btn');
  const downloadPause = $('download-pause-btn');
  const downloadResume = $('download-resume-btn');
  const downloadCancel = $('download-cancel-btn');
  const downloadRefresh = $('download-refresh-btn');
  const runtimePreset = $('runtime-preset');
  const runtimeFile = $('runtime-config-file');
  const runtimeClear = $('runtime-config-clear');
  const runtimeConfigPreset = $('runtime-config-preset');
  const diagnosticsModelSelect = $('diagnostics-model');
  const diagnosticsRun = $('diagnostics-run-btn');
  const diagnosticsVerify = $('diagnostics-verify-btn');
  const diagnosticsExport = $('diagnostics-export-btn');
  const exportModelBtn = $('export-model-btn');
  const unloadModelBtn = $('unload-model-btn');
  const clearMemoryBtn = $('clear-memory-btn');
  const storageInspectorRefresh = $('storage-inspector-refresh');
  const chatInput = $('chat-input');
  const sendBtn = $('send-btn');
  const stopBtn = $('stop-btn');
  const clearBtn = $('clear-btn');
  const temperatureInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const diffusionPrompt = $('diffusion-prompt');
  const diffusionNegative = $('diffusion-negative');
  const diffusionSteps = $('diffusion-steps');
  const diffusionGuidance = $('diffusion-guidance');
  const diffusionSeed = $('diffusion-seed');
  const diffusionWidth = $('diffusion-width');
  const diffusionHeight = $('diffusion-height');
  const diffusionRun = $('diffusion-run-btn');
  const diffusionClear = $('diffusion-clear-btn');

  document.querySelectorAll('.mode-tab').forEach((button) => {
    button.addEventListener('click', () => {
      const mode = button.dataset.mode || 'run';
      setUiMode(mode);
    });
  });

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

  regenManifestBtn?.addEventListener('click', () => {
    resetConvertStatus();
    handleRegenerateManifest().catch((error) => {
      updateConvertStatus(`Manifest error: ${error.message}`);
    });
  });

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

  storageInspectorRefresh?.addEventListener('click', () => {
    refreshStorageInspector();
  });

  runtimePreset?.addEventListener('change', () => {
    applySelectedRuntimePreset();
  });

  diagnosticsModelSelect?.addEventListener('change', () => {
    selectDiagnosticsModel(diagnosticsModelSelect.value || null);
  });

  runtimeFile?.addEventListener('change', () => {
    const file = runtimeFile.files?.[0] || null;
    handleRuntimeConfigFile(file);
  });

  runtimeConfigPreset?.addEventListener('change', () => {
    const presetId = runtimeConfigPreset.value || '';
    if (runtimeFile) {
      runtimeFile.value = '';
    }
    applyRuntimeConfigPreset(presetId);
  });

  runtimeClear?.addEventListener('click', () => {
    state.runtimeOverride = null;
    state.runtimeOverrideBase = null;
    state.runtimeOverrideLabel = null;
    state.runtimeSamplingOverride = null;
    state.runtimeSamplingLabel = null;
    runtimeFile.value = '';
    if (runtimeConfigPreset) {
      runtimeConfigPreset.value = '';
    }
    applySelectedRuntimePreset();
  });

  temperatureInput?.addEventListener('input', applySamplingOverrideFromInputs);
  topPInput?.addEventListener('input', applySamplingOverrideFromInputs);
  topKInput?.addEventListener('input', applySamplingOverrideFromInputs);

  diagnosticsRun?.addEventListener('click', () => handleDiagnosticsRun('run'));
  diagnosticsVerify?.addEventListener('click', () => handleDiagnosticsRun('verify'));
  diagnosticsExport?.addEventListener('click', exportDiagnosticsReport);
  exportModelBtn?.addEventListener('click', () => {
    resetExportStatus();
    exportActiveModel().catch((error) => {
      updateExportStatus(`Export error: ${error.message}`, 0);
    });
  });

  unloadModelBtn?.addEventListener('click', () => {
    unloadActivePipeline().catch((error) => {
      log.warn('DopplerDemo', `Unload failed: ${error.message}`);
    });
  });

  clearMemoryBtn?.addEventListener('click', () => {
    clearAllMemory().catch((error) => {
      log.warn('DopplerDemo', `Clear memory failed: ${error.message}`);
    });
  });

  sendBtn?.addEventListener('click', () => {
    handleChatSend().catch((error) => {
      log.warn('DopplerDemo', `Chat send failed: ${error.message}`);
    });
  });

  chatInput?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleChatSend().catch((error) => {
        log.warn('DopplerDemo', `Chat send failed: ${error.message}`);
      });
    }
  });

  stopBtn?.addEventListener('click', () => {
    stopChatGeneration();
  });

  clearBtn?.addEventListener('click', () => {
    clearChatMessages();
  });

  diffusionClear?.addEventListener('click', () => {
    if (diffusionPrompt) diffusionPrompt.value = '';
    if (diffusionNegative) diffusionNegative.value = '';
    if (diffusionSteps) diffusionSteps.value = '20';
    if (diffusionGuidance) diffusionGuidance.value = '7.5';
    if (diffusionSeed) diffusionSeed.value = '';
    if (diffusionWidth) diffusionWidth.value = '256';
    if (diffusionHeight) diffusionHeight.value = '256';
    handleDiffusionClear();
  });

  diffusionRun?.addEventListener('click', () => {
    handleDiffusionRun().catch((error) => {
      log.error('DopplerDemo', `Diffusion run failed: ${error.message}`);
      updateDiffusionStatus(`Error: ${error.message}`);
    });
  });

}

async function init() {
  setStatusIndicator('Initializing', 'info');
  populateModelPresets();
  populateRuntimeConfigPresets();
  setUiMode(state.uiMode);
  await refreshModelList();
  await refreshGpuInfo();
  await applySelectedRuntimePreset();
  await refreshDownloads();
  updateMemoryControls();
  resetExportStatus();
  startTelemetryLoop();
  setChatLoading(false);
  setChatGenerating(false);
  updateDiffusionStatus('Idle');
  updateStatusIndicator();
  bindUI();
}

init().catch((error) => {
  log.error('DopplerDemo', `Demo init failed: ${error.message}`);
});
