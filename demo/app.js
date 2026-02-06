import { log } from '../src/debug/index.js';
import { listPresets, createConverterConfig, detectPreset, resolvePreset } from '../src/config/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../src/config/schema/index.js';
import { formatBytes } from '../src/storage/quota.js';
import { listRegisteredModels, registerModel } from '../src/storage/registry.js';
import {
  openModelStore,
  loadManifestFromStore,
  loadShard,
  loadTensorsFromStore,
  saveManifest,
  saveTensorsToStore,
  loadTokenizerFromStore,
  loadTokenizerModelFromStore,
} from '../src/storage/shard-manager.js';
import { parseManifest, getManifest, setManifest, clearManifest, classifyTensorRole } from '../src/storage/rdrr-format.js';
import {
  convertModel,
  createRemoteModelSources,
  isConversionSupported,
} from '../src/browser/browser-converter.js';
import { buildManifestInference, inferEmbeddingOutputConfig } from '../src/converter/manifest-inference.js';
import { pickModelDirectory, pickModelFiles } from '../src/browser/file-picker.js';
import { createPipeline } from '../src/inference/pipeline.js';
import { initDevice, getDevice, getKernelCapabilities, getPlatformConfig, isWebGPUAvailable } from '../src/gpu/device.js';
import { captureMemorySnapshot } from '../src/loader/memory-monitor.js';
import { destroyBufferPool } from '../src/memory/buffer-pool.js';
import { DiagnosticsController } from './diagnostics-controller.js';
import { state } from './app/state.js';
import { $, setText, setHidden } from './app/dom.js';
import { formatAutoValue } from './app/format.js';
import { readOptionalNumber } from './app/input.js';
import {
  showProgressOverlay,
  hideProgressOverlay,
  updateProgressFromLoader,
} from './app/progress.js';
import {
  setStatusIndicator,
  updateStatusIndicator,
  clampPercent,
  hideErrorModal,
} from './app/ui.js';
import {
  updatePerformancePanel,
  updateMemoryPanel,
  updateMemoryControls,
  renderRunLog,
  recordRunLog,
} from './app/stats.js';
import {
  VLIW_DATASETS,
  ENERGY_DEMOS,
  DEFAULT_ENERGY_DEMO_ID,
  DEFAULT_RUNTIME_PRESET,
  RUNTIME_PRESET_REGISTRY,
} from './app/constants.js';
import {
  updateEnergyStatus,
  getEnergyDemoById,
  setEnergyMetricLabels,
  toggleEnergyProblemControls,
  syncEnergyDemoSelection,
  populateEnergyDemoSelect,
  applyEnergyDemoDefaults,
} from './app/energy/controls.js';
import {
  clearEnergyBoard,
  clearEnergyChart,
  clearEnergyKernelSummary,
  clearEnergyBundleView,
  renderEnergyBoard,
  renderEnergyVector,
  renderEnergyIntensityBoard,
  renderVliwKernelSummary,
  populateVliwBundleSelect,
  renderVliwBundleView,
  drawEnergyChart,
  updateEnergyStats,
} from './app/energy/render.js';
import {
  loadVliwDataset,
  applyWorkloadSpec,
  buildVliwDatasetFromSpecInput,
  sliceVliwDataset,
} from './app/energy/datasets.js';
import {
  resolveBaseSpec,
  runVliwSpecSearch,
  formatSpecSignature,
} from './app/energy/spec-search.js';
import {
  storeDiagnosticsSelection,
  syncDiagnosticsModeUI,
  getDiagnosticsDefaultSuite,
  getDiagnosticsRuntimeConfig,
  refreshDiagnosticsRuntimeConfig,
  syncDiagnosticsDefaultsForMode,
  clearDiagnosticsOutput,
  renderDiagnosticsOutput,
  updateDiagnosticsStatus,
  updateDiagnosticsReport,
  updateDiagnosticsGuidance,
  selectDiagnosticsModel,
  handleRuntimeConfigFile,
  applyRuntimeConfigPreset,
  applySelectedRuntimePreset,
} from './app/diagnostics/index.js';
import {
  normalizeModelType,
  isCompatibleModelType,
  isModeModelSelectable,
  getModeModelLabel,
  getModelTypeForId,
} from './app/models/utils.js';
import { updateStorageInfo, refreshStorageInspector } from './app/storage/inspector.js';
import {
  configureDownloadCallbacks,
  refreshDownloads,
  startDownload,
  pauseActiveDownload,
  resumeActiveDownload,
  cancelActiveDownload,
} from './app/downloads/index.js';

const controller = new DiagnosticsController({ log });

const PRIMARY_MODES = new Set(['run', 'diffusion', 'energy']);

function updateNavState(mode) {
  const activePrimary = PRIMARY_MODES.has(mode) ? mode : (state.lastPrimaryMode || 'run');
  document.querySelectorAll('.mode-tab').forEach((button) => {
    const isActive = button.dataset.mode === activePrimary;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
  document.querySelectorAll('.mode-tool').forEach((button) => {
    const target = button.dataset.mode;
    const isActive = target === 'models'
      ? mode === 'models'
      : (mode === 'diagnostics' || mode === 'kernels');
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
}


function cloneRuntimeConfig(config) {
  try {
    return structuredClone(config);
  } catch {
    return JSON.parse(JSON.stringify(config));
  }
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
  const previousMode = state.uiMode;
  state.uiMode = mode;
  if (PRIMARY_MODES.has(mode)) {
    state.lastPrimaryMode = mode;
  }
  app.dataset.mode = mode;
  updateNavState(mode);
  applyModeVisibility(mode);
  syncDiagnosticsModeUI(mode);
  updatePerformancePanel();
  renderRunLog();
  if (mode === 'models') {
    refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onModelsUpdated: refreshModelList,
    });
  }
  refreshModelList().catch((error) => {
    log.warn('DopplerDemo', `Model list refresh failed: ${error.message}`);
  });
  syncDiagnosticsDefaultsForMode(mode).catch((error) => {
    updateDiagnosticsStatus(`Diagnostics config error: ${error.message}`, true);
  });
  if (mode === 'energy') {
    syncEnergyDemoSelection();
  }
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

function updateRunStatus(message) {
  const status = $('run-output-status');
  if (!status) return;
  setText(status, message || 'Idle');
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

async function filterModelsForMode(models, mode) {
  if (!isModeModelSelectable(mode)) return models;
  const filtered = [];
  for (const model of models) {
    const modelId = model.modelId || model.id;
    if (!modelId) continue;
    const modelType = await getModelTypeForId(modelId);
    if (isCompatibleModelType(modelType, mode)) {
      filtered.push(model);
    }
  }
  return filtered;
}

async function registerDownloadedModel(modelId) {
  if (!modelId) return null;
  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) return null;
  const manifest = parseManifest(manifestText);
  const entry = {
    modelId,
    totalSize: manifest.totalSize,
    quantization: manifest.quantization,
    hashAlgorithm: manifest.hashAlgorithm,
    modelType: manifest.modelType,
  };
  if (manifest.modelId && manifest.modelId !== modelId) {
    entry.sourceModelId = manifest.modelId;
  }
  return registerModel(entry);
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

function updateSidebarLayout(models) {
  const panelGrid = $('panel-grid');
  if (!panelGrid) return;
  const hasModels = Array.isArray(models) && models.length > 0;
  panelGrid.dataset.layout = hasModels ? 'ready' : 'empty';
  if (!hasModels && state.uiMode !== 'models' && state.uiMode !== 'kernels' && state.uiMode !== 'diagnostics') {
    setUiMode('models');
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
  const filteredModels = await filterModelsForMode(models, state.uiMode);
  if (!filteredModels.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = `No ${getModeModelLabel(state.uiMode)} models`;
    modelSelect.appendChild(opt);
  } else {
    for (const model of filteredModels) {
      const opt = document.createElement('option');
      opt.value = model.modelId || model.id || '';
      opt.textContent = model.modelId || model.id || 'unknown';
      modelSelect.appendChild(opt);
    }
  }
  updateSidebarLayout(models);
  await updateStorageInfo();
  await syncModelForMode(state.uiMode);
  updateDiagnosticsGuidance();
  if (state.uiMode === 'models') {
    await refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onModelsUpdated: refreshModelList,
    });
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

function syncRunControls() {
  const runPrompt = $('run-prompt');
  const runGenerate = $('run-generate-btn');
  const runStop = $('run-stop-btn');
  const runClear = $('run-clear-btn');
  const temperatureInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const maxTokensInput = $('max-tokens-input');
  const disabled = state.runGenerating || state.runLoading;
  if (runPrompt) runPrompt.disabled = disabled;
  if (runGenerate) runGenerate.disabled = disabled;
  if (runClear) runClear.disabled = disabled;
  if (temperatureInput) temperatureInput.disabled = disabled;
  if (topPInput) topPInput.disabled = disabled;
  if (topKInput) topKInput.disabled = disabled;
  if (maxTokensInput) maxTokensInput.disabled = disabled;
  if (runStop) setHidden(runStop, !state.runGenerating);
}

function setRunGenerating(isGenerating) {
  state.runGenerating = Boolean(isGenerating);
  syncRunControls();
  updateStatusIndicator();
}

function setRunLoading(isLoading) {
  state.runLoading = Boolean(isLoading);
  syncRunControls();
  updateStatusIndicator();
}

function setRunAutoLabel(inputId, labelId, value, options) {
  const input = $(inputId);
  const label = $(labelId);
  if (!label) return;
  const hasOverride = input?.value != null && input.value !== '';
  const prefix = hasOverride ? 'default' : 'auto';
  label.textContent = `${prefix}: ${formatAutoValue(value, options)}`;
}

function updateRunAutoLabels() {
  const runtime = getRuntimeConfig();
  const sampling = runtime?.inference?.sampling ?? {};
  const batching = runtime?.inference?.batching ?? {};
  setRunAutoLabel('temperature-input', 'temperature-auto', sampling.temperature);
  setRunAutoLabel('top-p-input', 'top-p-auto', sampling.topP);
  setRunAutoLabel('top-k-input', 'top-k-auto', sampling.topK, { integer: true });
  setRunAutoLabel('max-tokens-input', 'max-tokens-auto', batching.maxTokens, { integer: true });
}

function formatCharCounter(value, maxLength) {
  const length = String(value || '').length;
  if (Number.isFinite(maxLength) && maxLength > 0) {
    return `${length}/${maxLength}`;
  }
  return String(length);
}

function updateDiffusionCharCounters() {
  const promptEl = $('diffusion-prompt');
  const negativeEl = $('diffusion-negative');
  const promptCountEl = $('diffusion-prompt-count');
  const negativeCountEl = $('diffusion-negative-count');

  if (promptCountEl) {
    const maxLength = promptEl?.maxLength ?? null;
    promptCountEl.textContent = formatCharCounter(promptEl?.value, maxLength);
  }
  if (negativeCountEl) {
    const maxLength = negativeEl?.maxLength ?? null;
    negativeCountEl.textContent = formatCharCounter(negativeEl?.value, maxLength);
  }
}

function buildRunGenerateOptions() {
  const temperature = readOptionalNumber($('temperature-input'));
  const topP = readOptionalNumber($('top-p-input'));
  const topK = readOptionalNumber($('top-k-input'), { integer: true });
  const maxTokens = readOptionalNumber($('max-tokens-input'), { integer: true });
  const options = {};
  if (temperature != null) {
    options.temperature = Math.max(0, temperature);
  }
  if (topP != null) {
    options.topP = Math.max(0, Math.min(1, topP));
  }
  if (topK != null) {
    options.topK = Math.max(0, topK);
  }
  if (maxTokens != null && maxTokens > 0) {
    options.maxTokens = Math.max(1, maxTokens);
  }
  return options;
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

async function ensureRunPipeline() {
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
  showProgressOverlay('Loading Model');
  setRunLoading(true);
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
    setRunLoading(false);
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
  state.lastDiffusionRequest = { ...request };

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
    if (result) {
      state.lastDiffusionRequest = {
        ...state.lastDiffusionRequest,
        width: result.width,
        height: result.height,
      };
    }
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




async function ensureEnergyPipeline() {
  const modelId = getSelectedModelId();
  if (!modelId) {
    throw new Error('Select a model before running energy.');
  }
  if (state.activePipeline && state.activeModelId === modelId) {
    return state.activePipeline;
  }
  if (state.activePipeline) {
    await unloadActivePipeline();
  }
  showProgressOverlay('Loading Energy Model');
  state.energyLoading = true;
  updateStatusIndicator();
  try {
    const pipeline = await loadPipelineFromStorage(modelId);
    state.activePipeline = pipeline;
    state.activeModelId = modelId;
    state.activePipelineModelId = modelId;
    if (pipeline?.manifest?.modelType) {
      state.modelTypeCache[modelId] = normalizeModelType(pipeline.manifest.modelType);
    }
    state.modeModelId.energy = modelId;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? null;
    updateMemoryControls();
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    return pipeline;
  } finally {
    hideProgressOverlay();
    state.energyLoading = false;
    updateStatusIndicator();
  }
}

async function handleEnergyRun() {
  if (state.energyGenerating || state.energyLoading) return;
  const demo = getEnergyDemoById(state.energyDemoId) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
  const problem = demo?.problem || 'quintel';
  const size = readOptionalNumber($('energy-quintel-size'), { integer: true });
  const displayThreshold = readOptionalNumber($('energy-quintel-threshold'));
  const countTarget = readOptionalNumber($('energy-quintel-count-target'), { integer: true });
  const mirrorX = $('energy-rule-mirror-x')?.checked ?? false;
  const mirrorY = $('energy-rule-mirror-y')?.checked ?? false;
  const diagonal = $('energy-rule-diagonal')?.checked ?? false;
  const countRule = $('energy-rule-count')?.checked ?? false;
  const symmetryWeight = readOptionalNumber($('energy-weight-symmetry'));
  const countWeight = readOptionalNumber($('energy-weight-count'));
  const binarizeWeight = readOptionalNumber($('energy-weight-binarize'));
  const initMode = $('energy-init-mode')?.value || undefined;
  const initSeed = readOptionalNumber($('energy-init-seed'), { integer: true });
  const initScale = readOptionalNumber($('energy-init-scale'));
  const steps = readOptionalNumber($('energy-steps'), { integer: true });
  const stepSize = readOptionalNumber($('energy-step-size'));
  const gradientScale = readOptionalNumber($('energy-gradient-scale'));
  const convergenceThreshold = readOptionalNumber($('energy-convergence'));

  const request = {
    problem,
    initMode,
    seed: initSeed,
    initScale,
    steps,
    stepSize,
    gradientScale,
    convergenceThreshold,
  };
  let vliwRun = null;
  let vliwBundleLimit = null;

  if (problem !== 'vliw') {
    state.energyVliw = null;
    state.energyVliwTasks = null;
    state.energyVliwCaps = null;
    state.energyVliwOps = null;
    state.energyVliwBundle = null;
    state.energyVliwBundleLimit = null;
    state.energyVliwMeta = null;
    state.energyVliwDatasetId = null;
    state.energyVliwSpecSearch = null;
    state.lastEnergyResult = null;
    clearEnergyKernelSummary();
    clearEnergyBundleView();
  }

  if (problem === 'quintel') {
    const quintelRules = {
      mirrorX,
      mirrorY,
      diagonal,
      count: countRule,
      center: false,
    };
    const quintel = {
      rules: quintelRules,
    };
    if (size != null) quintel.size = size;
    if (Number.isFinite(countTarget)) quintel.countTarget = countTarget;
    const weights = {};
    if (Number.isFinite(symmetryWeight)) weights.symmetry = symmetryWeight;
    if (Number.isFinite(countWeight)) weights.count = countWeight;
    if (Number.isFinite(binarizeWeight)) weights.binarize = binarizeWeight;
    if (Object.keys(weights).length) quintel.weights = weights;
    request.quintel = quintel;
  }

  if (problem === 'vliw') {
    let datasetId = $('energy-vliw-dataset')?.value || 'vliw-simd-frozen';
    const selectedDatasetId = datasetId;
    const specText = $('energy-vliw-spec')?.value?.trim() || '';
    const bundleLimit = readOptionalNumber($('energy-vliw-bundle-limit'), { integer: true });
    const restarts = readOptionalNumber($('energy-vliw-restarts'), { integer: true });
    const tempStart = readOptionalNumber($('energy-vliw-temp-start'));
    const tempDecay = readOptionalNumber($('energy-vliw-temp-decay'));
    const mutationCount = readOptionalNumber($('energy-vliw-mutation'), { integer: true });
    const mlpHidden = readOptionalNumber($('energy-vliw-mlp-hidden'), { integer: true });
    const mlpLr = readOptionalNumber($('energy-vliw-mlp-lr'));
    const demoDefaults = demo?.defaults ?? {};
    const policy = demoDefaults.vliw?.policy;
    const jitter = demoDefaults.vliw?.jitter;
    const mlpDefaults = demoDefaults.vliw?.mlp ?? null;
    const vliwMode = $('energy-vliw-mode')?.value || demoDefaults.vliw?.mode || 'parity';
    const scoreMode = $('energy-vliw-score-mode')?.value || demoDefaults.vliw?.scoreMode || 'auto';
    const schedulerPolicies = Array.isArray(demoDefaults.vliw?.schedulerPolicies)
      ? demoDefaults.vliw.schedulerPolicies
      : null;
    const schedulerRestarts = Number.isFinite(demoDefaults.vliw?.schedulerRestarts)
      ? demoDefaults.vliw.schedulerRestarts
      : null;
    const capsSource = vliwMode === 'parity' ? 'slot_limits' : 'spec';
    const mlpConfig = policy === 'mlp'
      ? {
        ...(Number.isFinite(mlpHidden) ? { hiddenSize: mlpHidden } : {}),
        ...(Number.isFinite(mlpLr) ? { lr: mlpLr } : {}),
        ...(!Number.isFinite(mlpHidden) && Number.isFinite(mlpDefaults?.hiddenSize) ? { hiddenSize: mlpDefaults.hiddenSize } : {}),
        ...(!Number.isFinite(mlpLr) && Number.isFinite(mlpDefaults?.lr) ? { lr: mlpDefaults.lr } : {}),
      }
      : null;
    const mlp = mlpConfig && Object.keys(mlpConfig).length ? mlpConfig : null;
    const vliwSearch = {
      restarts,
      temperatureStart: tempStart,
      temperatureDecay: tempDecay,
      mutationCount,
      ...(policy ? { policy } : {}),
      ...(mlp ? { mlp } : {}),
      ...(Number.isFinite(jitter) ? { jitter } : {}),
      ...(vliwMode ? { mode: vliwMode } : {}),
      ...(scoreMode ? { scoreMode } : {}),
      ...(capsSource ? { capsSource } : {}),
      ...(schedulerPolicies ? { schedulerPolicies } : {}),
      ...(Number.isFinite(schedulerRestarts) ? { schedulerRestarts } : {}),
    };
    const constraintDefaults = demoDefaults.vliw?.specSearch?.constraints || {};
    const specSearchDefaults = demoDefaults.vliw?.specSearch || {};
    const frozenWorkloadSpec = VLIW_DATASETS[selectedDatasetId]?.spec || null;
    const specSearch = {
      enabled: $('energy-vliw-spec-search')?.checked ?? false,
      restarts: readOptionalNumber($('energy-vliw-spec-restarts'), { integer: true }),
      steps: readOptionalNumber($('energy-vliw-spec-steps'), { integer: true }),
      temperatureStart: readOptionalNumber($('energy-vliw-spec-temp-start')),
      temperatureDecay: readOptionalNumber($('energy-vliw-spec-temp-decay')),
      mutationCount: readOptionalNumber($('energy-vliw-spec-mutation'), { integer: true }),
      seed: readOptionalNumber($('energy-vliw-spec-seed'), { integer: true }),
      penaltyGate: readOptionalNumber($('energy-vliw-spec-penalty')),
      cycleLambda: readOptionalNumber($('energy-vliw-spec-lambda')),
      innerSteps: readOptionalNumber($('energy-vliw-spec-inner-steps'), { integer: true }),
      lbPenalty: specSearchDefaults.lbPenalty,
      targetCycles: specSearchDefaults.targetCycles,
      scoreMode: specSearchDefaults.scoreMode || scoreMode,
      constraints: {
        mode: vliwMode === 'relaxed' ? 'relaxed' : 'parity',
        fallbackCycles: constraintDefaults.fallbackCycles ?? 10000,
      },
    };
    if (specSearch.enabled) {
      vliwBundleLimit = vliwMode === 'parity' ? 0 : bundleLimit;
      let baseSpecInput = null;
      let baseDataset = null;
      if (specText) {
        try {
          baseSpecInput = JSON.parse(specText);
        } catch (error) {
          throw new Error(`Spec JSON parse error: ${error.message}`);
        }
      } else {
        baseDataset = await loadVliwDataset(datasetId);
        baseSpecInput = baseDataset?.spec ?? null;
      }
      if (frozenWorkloadSpec) {
        baseSpecInput = applyWorkloadSpec(baseSpecInput, frozenWorkloadSpec);
      }
      const baseSpec = resolveBaseSpec(baseSpecInput);
      vliwRun = {
        mode: 'spec-search',
        baseSpec,
        baseDataset,
        datasetId,
        bundleLimit,
        specSearch,
        vliwSearch,
      };
    } else {
      let dataset = null;
      if (specText) {
        let specInput = null;
        try {
          specInput = JSON.parse(specText);
        } catch (error) {
          throw new Error(`Spec JSON parse error: ${error.message}`);
        }
        dataset = await buildVliwDatasetFromSpecInput(specInput, specText, {
          mode: vliwMode,
          capsMode: capsSource,
          workloadSpec: frozenWorkloadSpec,
          includeOps: true,
        });
        datasetId = 'vliw-generated';
      } else {
        dataset = await loadVliwDataset(datasetId, { includeOps: true });
      }
      if (Number.isFinite(dataset?.spec?.sched_seed)) {
        vliwSearch.schedulerSeed = dataset.spec.sched_seed;
      }
      if (Number.isFinite(dataset?.spec?.sched_jitter)) {
        vliwSearch.schedulerJitter = dataset.spec.sched_jitter;
      }
      if (Number.isFinite(dataset?.spec?.sched_restarts)) {
        vliwSearch.schedulerRestarts = dataset.spec.sched_restarts;
      }
      const effectiveBundleLimit = vliwSearch.mode === 'parity' ? 0 : bundleLimit;
      vliwBundleLimit = effectiveBundleLimit;
      const sliced = sliceVliwDataset(dataset, effectiveBundleLimit);
      state.energyVliwMeta = {
        label: dataset.label || VLIW_DATASETS[datasetId]?.label || datasetId,
        bundleCount: sliced.bundleCount ?? dataset.bundleCount,
        taskCount: sliced.taskCount ?? sliced.tasks?.length ?? dataset.taskCount,
        baselineCycles: dataset.baselineCycles ?? dataset.bundleCount,
        dagHash: dataset.dag?.hash ?? dataset.dagHash,
        dependencyModel: dataset.dependencyModel ?? null,
        spec: dataset.spec ?? null,
      };
      state.energyVliwTasks = sliced.tasks;
      state.energyVliwCaps = sliced.caps;
      state.energyVliwOps = dataset.ops ?? null;
      state.energyVliwDatasetId = datasetId;
      state.energyVliwBundleLimit = vliwBundleLimit;
      request.vliw = {
        tasks: sliced.tasks,
        caps: sliced.caps,
        dependencyModel: dataset.dependencyModel ?? null,
        search: vliwSearch,
      };
    }
  }
  state.lastEnergyRequest = {
    size,
    displayThreshold,
  };

  updateEnergyStatus('Preparing...');
  state.energyGenerating = true;
  updateStatusIndicator();
  try {
    const pipeline = await ensureEnergyPipeline();
    if (!pipeline.generate) {
      throw new Error('Selected model does not support energy generation.');
    }
    if (!pipeline.manifest || pipeline.manifest.modelType !== 'energy') {
      throw new Error('Selected model is not an energy model.');
    }
    updateEnergyStatus('Running...');
    let result = null;
    let specSearchSummary = null;
    if (problem === 'vliw' && vliwRun?.mode === 'spec-search') {
      updateEnergyStatus('Spec search (Layer 0)...');
      const specSearchResult = await runVliwSpecSearch({
        pipeline,
        baseSpec: vliwRun.baseSpec,
        innerRequestBase: request,
        vliwSearch: vliwRun.vliwSearch,
        bundleLimit: vliwRun.bundleLimit,
        specSearch: vliwRun.specSearch,
      });
      result = specSearchResult.result;
      const dataset = specSearchResult.dataset;
      const sliced = specSearchResult.sliced;
      state.energyVliwMeta = {
        label: dataset.label || 'Spec search (Layer 0)',
        bundleCount: sliced.bundleCount ?? dataset.bundleCount,
        taskCount: sliced.taskCount ?? sliced.tasks?.length ?? dataset.taskCount,
        baselineCycles: dataset.baselineCycles ?? dataset.bundleCount,
        dagHash: dataset.dag?.hash ?? dataset.dagHash,
        dependencyModel: dataset.dependencyModel ?? null,
        spec: dataset.spec ?? specSearchResult.bestSpec ?? null,
      };
      state.energyVliwTasks = sliced.tasks;
      state.energyVliwCaps = sliced.caps;
      state.energyVliwOps = dataset.ops ?? null;
      state.energyVliwDatasetId = vliwRun.datasetId || datasetId || 'vliw-spec-search';
      state.energyVliwBundleLimit = vliwBundleLimit;
      specSearchSummary = {
        restarts: specSearchResult.restarts,
        steps: specSearchResult.steps,
        cycleLambda: specSearchResult.cycleLambda,
        penaltyGate: specSearchResult.penaltyGate,
        fallbackCycles: specSearchResult.fallbackCycles,
        lbPenalty: specSearchResult.lbPenalty,
        targetCycles: specSearchResult.targetCycles,
        scoreMode: specSearchResult.scoreMode,
        constraintMode: specSearchResult.constraintMode,
        scheduler: specSearchResult.scheduler,
        bestCycles: specSearchResult.bestCycles,
        bestPenalty: specSearchResult.bestPenalty,
        bestEnergy: specSearchResult.bestEnergy,
        bestSpecSignature: specSearchResult.bestSpec ? formatSpecSignature(specSearchResult.bestSpec) : null,
        candidates: specSearchResult.candidates.map((candidate) => ({
          cycles: candidate.cycles,
          penalty: candidate.penalty,
          signature: formatSpecSignature(candidate.spec),
        })),
      };
      state.energyVliwSpecSearch = specSearchSummary;
    } else {
      result = await pipeline.generate(request);
    }
    if (problem === 'vliw') {
      state.energyVliw = {
        schedule: result?.schedule || null,
        taskMeta: result?.taskMeta || null,
      };
      if (!vliwRun || vliwRun.mode !== 'spec-search') {
        state.energyVliwSpecSearch = null;
      }
      const candidates = Array.isArray(result?.candidates) ? result.candidates.slice() : [];
      candidates.sort((a, b) => a.cycles - b.cycles);
      const summary = {
        bestCycles: result?.metrics?.cycles ?? null,
        utilization: result?.metrics?.utilization ?? null,
        candidates: candidates.slice(0, 6),
        baseline: result?.baseline ?? null,
        specSearch: specSearchSummary,
        scheduler: result?.scheduler ?? null,
        schedulerPolicy: result?.schedulerPolicy ?? null,
        schedulerPolicies: result?.schedulerPolicies ?? null,
        scoreMode: result?.scoreMode ?? null,
        engineOrder: result?.engineOrder ?? null,
        capsSource: result?.capsSource ?? null,
        bundleLimit: vliwBundleLimit,
        mode: result?.mode ?? null,
        mlpStats: result?.mlpStats ?? null,
      };
      renderVliwKernelSummary(summary, state.energyVliwMeta);
      const bundleCount = state.energyVliwMeta?.bundleCount;
      populateVliwBundleSelect(bundleCount);
      const bundleSelect = $('energy-vliw-bundle-select');
      if (bundleSelect) {
        const selected = Number.isFinite(state.energyVliwBundle)
          ? state.energyVliwBundle
          : null;
        if (selected != null && selected >= 0) {
          bundleSelect.value = String(selected);
        }
        state.energyVliwBundle = selected;
      }
      renderVliwBundleView(state.energyVliw, state.energyVliwBundle);
    }
    state.lastEnergyResult = result;
    if (result?.shape) {
      state.lastEnergyRequest = {
        shape: result.shape,
        size: result.shape[0],
        displayThreshold,
      };
    }
    drawEnergyChart(result?.energyHistory || []);
    updateEnergyStats(result);
    renderEnergyBoard(result?.state, result?.shape ?? size, displayThreshold);
    state.lastInferenceStats = pipeline.getStats?.() ?? null;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? state.lastMemoryStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`, 'energy');
    }
    updateEnergyStatus('Complete');
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
  } catch (error) {
    log.error('DopplerDemo', `Energy run failed: ${error.message}`);
    updateEnergyStatus(`Error: ${error.message}`);
  } finally {
    state.energyGenerating = false;
    updateStatusIndicator();
  }
}

function handleEnergyClear() {
  clearEnergyChart();
  clearEnergyBoard();
  updateEnergyStats(null);
  updateEnergyStatus('Idle');
  state.energyVliw = null;
  state.energyVliwTasks = null;
  state.energyVliwCaps = null;
  state.energyVliwOps = null;
  state.energyVliwBundle = null;
  state.energyVliwBundleLimit = null;
  state.energyVliwMeta = null;
  state.energyVliwDatasetId = null;
  state.energyVliwSpecSearch = null;
  state.lastEnergyResult = null;
}

async function handleRunGenerate() {
  if (state.runGenerating || state.runLoading) return;
  const promptEl = $('run-prompt');
  const outputEl = $('run-output');
  const prompt = promptEl?.value?.trim() || '';
  if (!prompt) {
    updateRunStatus('Enter a prompt to generate.');
    return;
  }

  updateRunStatus('Preparing...');
  let pipeline;
  try {
    pipeline = await ensureRunPipeline();
    if (pipeline?.manifest?.modelType === 'diffusion' || pipeline?.manifest?.modelType === 'energy') {
      throw new Error('Selected model is not a text model.');
    }
  } catch (error) {
    updateRunStatus(`Error: ${error.message}`);
    return;
  }

  const controller = new AbortController();
  state.runAbortController = controller;
  setRunGenerating(true);
  updateRunStatus('Generating...');
  if (outputEl) outputEl.textContent = '';

  const options = buildRunGenerateOptions();
  let output = '';
  let tokenCount = 0;
  const start = performance.now();
  let firstTokenAt = null;

  try {
    for await (const token of pipeline.generate(prompt, {
      ...options,
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
      if (outputEl) outputEl.textContent = output;
    }
    updateRunStatus(controller.signal.aborted ? 'Stopped' : 'Complete');
  } catch (error) {
    if (controller.signal.aborted) {
      updateRunStatus('Stopped');
    } else {
      updateRunStatus(`Error: ${error.message}`);
    }
  } finally {
    const elapsed = Math.max(1, performance.now() - start);
    const tokensPerSec = tokenCount > 0 ? Number(((tokenCount / elapsed) * 1000).toFixed(2)) : null;
    state.lastMetrics = {
      ...(state.lastMetrics || {}),
      tokensPerSec,
      liveTokensPerSec: null,
    };
    state.lastMemoryStats = pipeline?.getMemoryStats?.() ?? state.lastMemoryStats;
    state.lastInferenceStats = pipeline?.getStats?.() ?? state.lastInferenceStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    setRunGenerating(false);
    state.runAbortController = null;
  }
}

function stopRunGeneration() {
  if (state.runAbortController) {
    state.runAbortController.abort();
  }
}

function handleRunClear() {
  const promptEl = $('run-prompt');
  const outputEl = $('run-output');
  if (promptEl) promptEl.value = '';
  if (outputEl) outputEl.textContent = '';
  updateRunStatus('Idle');
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

  let telemetryInFlight = false;
  const tick = async () => {
    if (telemetryInFlight) return;
    telemetryInFlight = true;
    try {
      const now = Date.now();
      if (now - state.lastStorageRefresh > 15000) {
        state.lastStorageRefresh = now;
        await updateStorageInfo();
      }
      const snapshot = captureMemorySnapshot();
      updateMemoryPanel(snapshot);
      updatePerformancePanel(snapshot);
    } catch (error) {
      log.warn('DopplerDemo', `Telemetry update failed: ${error.message}`);
    } finally {
      telemetryInFlight = false;
    }
  };

  state.uiIntervalId = setInterval(() => {
    void tick();
  }, 1000);
  void tick();
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

function populateRuntimePresetSelect(select, entries, fallbackValue) {
  if (!select) return;
  const previous = select.value;
  select.innerHTML = '';
  for (const entry of entries) {
    const opt = document.createElement('option');
    opt.value = entry.id;
    opt.textContent = entry.label;
    select.appendChild(opt);
  }
  const target = previous || fallbackValue;
  if (target !== undefined && entries.some((entry) => entry.id === target)) {
    select.value = target;
    return;
  }
  if (entries.length > 0) {
    select.value = entries[0].id;
  }
}

function populateRuntimePresetSelects() {
  const baseSelect = $('runtime-preset');
  const overrideSelect = $('runtime-config-preset');
  const baseEntries = RUNTIME_PRESET_REGISTRY.filter((entry) => entry.base);
  const overrideEntries = RUNTIME_PRESET_REGISTRY.filter((entry) => entry.override);
  populateRuntimePresetSelect(baseSelect, baseEntries, DEFAULT_RUNTIME_PRESET);
  populateRuntimePresetSelect(overrideSelect, overrideEntries, '');
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
    const modelConfig = rawConfig?.text_config ?? rawConfig ?? {};
    const hiddenSize = modelConfig.hidden_size ?? modelConfig.n_embd ?? modelConfig.d_model ?? modelConfig.model_dim ?? null;
    const numHeads = modelConfig.num_attention_heads ?? modelConfig.n_head ?? modelConfig.num_heads ?? null;
    const derivedHeadDim = (Number.isFinite(hiddenSize) && Number.isFinite(numHeads) && numHeads > 0)
      ? hiddenSize / numHeads
      : null;
    const configHeadDim = Number.isFinite(rawConfig.head_dim) ? rawConfig.head_dim : null;
    const manifestHeadDim = (
      manifest.architecture
      && typeof manifest.architecture === 'object'
      && Number.isFinite(manifest.architecture.headDim)
    )
      ? manifest.architecture.headDim
      : null;
    const headDim = configHeadDim
      ?? manifestHeadDim
      ?? (Number.isFinite(derivedHeadDim) && Math.floor(derivedHeadDim) === derivedHeadDim ? derivedHeadDim : null);
    if (!headDim) {
      throw new Error('Missing headDim in manifest config (head_dim or hidden_size/num_attention_heads).');
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
    if (modelId) {
      delete state.modelTypeCache[modelId];
    }
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

async function handleDiagnosticsRun(mode) {
  const suiteSelect = $('diagnostics-suite');
  const modelSelect = $('diagnostics-model');
  const presetSelect = $('runtime-preset');
  const suite = suiteSelect?.value || getDiagnosticsDefaultSuite(state.uiMode);
  const modelId = modelSelect?.value || null;
  const runtimePreset = presetSelect?.value || DEFAULT_RUNTIME_PRESET;
  const captureOutput = runtimePreset === 'modes/debug';
  const previousRuntime = cloneRuntimeConfig(getRuntimeConfig());
  let runtimeConfig = state.diagnosticsRuntimeConfig;

  updateDiagnosticsStatus(`${mode === 'verify' ? 'Verifying' : 'Running'} ${suite}...`);
  updateDiagnosticsReport('');
  clearDiagnosticsOutput();
  try {
    if (!runtimeConfig || state.diagnosticsRuntimePresetId !== runtimePreset) {
      runtimeConfig = await refreshDiagnosticsRuntimeConfig(runtimePreset);
    }
    if (mode === 'verify') {
      await controller.verifySuite(null, {
        suite,
        runtimeConfig,
      });
      updateDiagnosticsStatus('Verified');
      clearDiagnosticsOutput();
      return;
    }

    if (state.activePipeline) {
      await unloadActivePipeline();
    }

    const options = {
      suite,
      runtimePreset,
      modelId,
      runtimeConfig,
      captureOutput,
    };
    const result = await controller.runSuite(
      modelId ? { sources: { browser: { id: modelId } } } : null,
      { ...options, keepPipeline: true }
    );
    state.lastReport = result.report;
    state.lastReportInfo = result.reportInfo;
    state.lastMetrics = result.metrics ?? null;
    state.lastDiagnosticsSuite = result.suite;
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
    if (result.suite === 'diffusion' && result.metrics) {
      state.lastDiffusionRequest = {
        width: result.metrics.width,
        height: result.metrics.height,
        steps: result.metrics.steps,
      };
    }
    if (result.suite === 'energy' && result.metrics) {
      const shape = Array.isArray(result.metrics.shape) ? result.metrics.shape : null;
      if (shape) {
        state.lastEnergyRequest = {
          shape,
          height: shape[0],
          width: shape[1],
          channels: shape[2],
        };
      }
      if (Array.isArray(result.metrics.energyHistory)) {
        drawEnergyChart(result.metrics.energyHistory);
      }
      updateEnergyStats({
        steps: result.metrics.steps,
        energy: result.metrics.energy,
        dtype: result.metrics.dtype,
        shape,
        stateStats: result.metrics.stateStats,
      });
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
    renderDiagnosticsOutput(result, suite, captureOutput);
  } catch (error) {
    updateDiagnosticsStatus(error.message, true);
    clearDiagnosticsOutput();
  } finally {
    setRuntimeConfig(previousRuntime);
    updateRunAutoLabels();
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

function serializeTypedArray(value) {
  if (!value) return null;
  if (ArrayBuffer.isView(value)) return Array.from(value);
  return value;
}

function serializeSchedule(schedule) {
  if (!schedule) return null;
  return {
    slotAssignments: serializeTypedArray(schedule.slotAssignments),
    slotEngines: Array.isArray(schedule.slotEngines) ? schedule.slotEngines.slice() : schedule.slotEngines,
    slotIndices: Array.isArray(schedule.slotIndices) ? schedule.slotIndices.slice() : schedule.slotIndices,
  };
}

function serializeOps(ops) {
  if (!Array.isArray(ops)) return null;
  return ops.map((op) => ({
    id: op?.id ?? null,
    engine: op?.engine ?? null,
    slot: Array.isArray(op?.slot) ? op.slot.slice() : op?.slot ?? null,
    offloadable: !!op?.offloadable,
    meta: op?.meta ?? null,
  }));
}

function exportEnergyRun() {
  if (!state.lastEnergyResult || !state.energyVliwTasks || !state.energyVliwCaps) {
    updateEnergyStatus('No VLIW run available to export.');
    return;
  }
  const payload = {
    timestamp: new Date().toISOString(),
    problem: 'vliw',
    mode: state.lastEnergyResult.mode ?? null,
    scoreMode: state.lastEnergyResult.scoreMode ?? null,
    scheduler: state.lastEnergyResult.scheduler ?? null,
    schedulerPolicy: state.lastEnergyResult.schedulerPolicy ?? null,
    schedulerPolicies: state.lastEnergyResult.schedulerPolicies ?? null,
    engineOrder: state.lastEnergyResult.engineOrder ?? null,
    capsSource: state.lastEnergyResult.capsSource ?? null,
    bundleLimit: state.energyVliwBundleLimit ?? null,
    dataset: {
      id: state.energyVliwDatasetId ?? null,
      label: state.energyVliwMeta?.label ?? null,
      dagHash: state.energyVliwMeta?.dagHash ?? null,
      bundleCount: state.energyVliwMeta?.bundleCount ?? null,
      taskCount: state.energyVliwMeta?.taskCount ?? null,
      baselineCycles: state.energyVliwMeta?.baselineCycles ?? null,
      dependencyModel: state.energyVliwMeta?.dependencyModel ?? null,
      spec: state.energyVliwMeta?.spec ?? null,
    },
    tasks: state.energyVliwTasks,
    ops: serializeOps(state.energyVliwOps),
    caps: state.energyVliwCaps,
    result: {
      metrics: state.lastEnergyResult.metrics ?? null,
      baseline: state.lastEnergyResult.baseline ?? null,
      candidates: state.lastEnergyResult.candidates ?? null,
      energyHistory: state.lastEnergyResult.energyHistory ?? null,
      schedule: serializeSchedule(state.lastEnergyResult.schedule),
    },
    specSearch: state.energyVliwSpecSearch ?? null,
  };
  const filename = `doppler-energy-export-${payload.timestamp.replace(/[:]/g, '-')}.json`;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
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
  const errorModal = $('error-modal');
  const errorClose = $('error-close');
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
  const diagnosticsSuite = $('diagnostics-suite');
  const diagnosticsRun = $('diagnostics-run-btn');
  const diagnosticsVerify = $('diagnostics-verify-btn');
  const diagnosticsExport = $('diagnostics-export-btn');
  const exportModelBtn = $('export-model-btn');
  const unloadModelBtn = $('unload-model-btn');
  const clearMemoryBtn = $('clear-memory-btn');
  const storageInspectorRefresh = $('storage-inspector-refresh');
  const runPrompt = $('run-prompt');
  const runGenerate = $('run-generate-btn');
  const runStop = $('run-stop-btn');
  const runClear = $('run-clear-btn');
  const temperatureInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const maxTokensInput = $('max-tokens-input');
  const diffusionPrompt = $('diffusion-prompt');
  const diffusionNegative = $('diffusion-negative');
  const diffusionSteps = $('diffusion-steps');
  const diffusionGuidance = $('diffusion-guidance');
  const diffusionSeed = $('diffusion-seed');
  const diffusionWidth = $('diffusion-width');
  const diffusionHeight = $('diffusion-height');
  const diffusionRun = $('diffusion-run-btn');
  const diffusionClear = $('diffusion-clear-btn');
  const energyDemoSelect = $('energy-demo-select');
  const energyRun = $('energy-run-btn');
  const energyExport = $('energy-export-btn');
  const energyClear = $('energy-clear-btn');

  errorClose?.addEventListener('click', () => hideErrorModal());
  errorModal?.addEventListener('click', (event) => {
    if (event.target === errorModal) {
      hideErrorModal();
    }
  });

  document.querySelectorAll('.mode-tab').forEach((button) => {
    button.addEventListener('click', () => {
      const mode = button.dataset.mode || 'run';
      setUiMode(mode);
    });
  });

  document.querySelectorAll('.mode-tool').forEach((button) => {
    button.addEventListener('click', () => {
      const mode = button.dataset.mode || 'diagnostics';
      setUiMode(mode);
    });
  });

  document.querySelectorAll('.diagnostics-mode-tab').forEach((button) => {
    button.addEventListener('click', () => {
      const mode = button.dataset.diagnosticsMode || 'diagnostics';
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
    refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onModelsUpdated: refreshModelList,
    });
  });

  runtimePreset?.addEventListener('change', () => {
    const mode = state.uiMode;
    storeDiagnosticsSelection(mode, { preset: runtimePreset.value || DEFAULT_RUNTIME_PRESET });
    if (runtimePreset.value !== 'modes/debug') {
      clearDiagnosticsOutput();
    }
    applySelectedRuntimePreset();
  });

  diagnosticsModelSelect?.addEventListener('change', () => {
    selectDiagnosticsModel(diagnosticsModelSelect.value || null);
  });

  diagnosticsSuite?.addEventListener('change', () => {
    const mode = state.uiMode;
    storeDiagnosticsSelection(mode, { suite: diagnosticsSuite.value || getDiagnosticsDefaultSuite(mode) });
    updateDiagnosticsGuidance();
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
    if (runtimeFile) {
      runtimeFile.value = '';
    }
    if (runtimeConfigPreset) {
      runtimeConfigPreset.value = '';
    }
    applySelectedRuntimePreset();
  });

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

  runGenerate?.addEventListener('click', () => {
    handleRunGenerate().catch((error) => {
      log.warn('DopplerDemo', `Run generate failed: ${error.message}`);
    });
  });

  runPrompt?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      handleRunGenerate().catch((error) => {
        log.warn('DopplerDemo', `Run generate failed: ${error.message}`);
      });
    }
  });

  runStop?.addEventListener('click', () => {
    stopRunGeneration();
  });

  runClear?.addEventListener('click', () => {
    handleRunClear();
  });

  temperatureInput?.addEventListener('input', updateRunAutoLabels);
  topPInput?.addEventListener('input', updateRunAutoLabels);
  topKInput?.addEventListener('input', updateRunAutoLabels);
  maxTokensInput?.addEventListener('input', updateRunAutoLabels);
  diffusionPrompt?.addEventListener('input', updateDiffusionCharCounters);
  diffusionNegative?.addEventListener('input', updateDiffusionCharCounters);

  diffusionClear?.addEventListener('click', () => {
    if (diffusionPrompt) diffusionPrompt.value = '';
    if (diffusionNegative) diffusionNegative.value = '';
    if (diffusionSteps) diffusionSteps.value = '20';
    if (diffusionGuidance) diffusionGuidance.value = '7.5';
    if (diffusionSeed) diffusionSeed.value = '';
    if (diffusionWidth) diffusionWidth.value = '256';
    if (diffusionHeight) diffusionHeight.value = '256';
    updateDiffusionCharCounters();
    handleDiffusionClear();
  });

  diffusionRun?.addEventListener('click', () => {
    handleDiffusionRun().catch((error) => {
      log.error('DopplerDemo', `Diffusion run failed: ${error.message}`);
      updateDiffusionStatus(`Error: ${error.message}`);
    });
  });

  energyDemoSelect?.addEventListener('change', () => {
    const demoId = energyDemoSelect.value || DEFAULT_ENERGY_DEMO_ID;
    const demo = getEnergyDemoById(demoId);
    if (!demo) return;
    state.energyDemoId = demo.id;
    setText($('energy-demo-description'), demo.description || '');
    setEnergyMetricLabels(demo.problem || 'quintel');
    toggleEnergyProblemControls(demo.problem || 'quintel');
    applyEnergyDemoDefaults(demo);
  });

  energyClear?.addEventListener('click', () => {
    const demoId = state.energyDemoId || DEFAULT_ENERGY_DEMO_ID;
    const demo = getEnergyDemoById(demoId) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
    if (demo) {
      applyEnergyDemoDefaults(demo);
    }
    handleEnergyClear();
  });

  energyRun?.addEventListener('click', () => {
    handleEnergyRun().catch((error) => {
      log.error('DopplerDemo', `Energy run failed: ${error.message}`);
      updateEnergyStatus(`Error: ${error.message}`);
    });
  });

  energyExport?.addEventListener('click', () => {
    exportEnergyRun();
  });

  const energyBundleSelect = $('energy-vliw-bundle-select');
  energyBundleSelect?.addEventListener('change', () => {
    const value = energyBundleSelect.value;
    const bundle = value === '' ? null : Number.parseInt(value, 10);
    state.energyVliwBundle = Number.isFinite(bundle) ? bundle : null;
    renderVliwBundleView(state.energyVliw, state.energyVliwBundle);
  });

  updateRunAutoLabels();
  updateDiffusionCharCounters();
}

async function init() {
  setStatusIndicator('Initializing', 'info');
  bindUI();
  configureDownloadCallbacks({
    onModelRegistered: registerDownloadedModel,
    onModelsUpdated: refreshModelList,
  });
  populateModelPresets();
  populateRuntimePresetSelects();
  populateEnergyDemoSelect();
  setUiMode(state.uiMode);
  await refreshModelList();
  await refreshGpuInfo();
  await refreshDownloads();
  updateMemoryControls();
  resetExportStatus();
  startTelemetryLoop();
  setRunLoading(false);
  setRunGenerating(false);
  updateRunStatus('Idle');
  updateDiffusionStatus('Idle');
  updateEnergyStatus('Idle');
  updateStatusIndicator();
}

init().catch((error) => {
  log.error('DopplerDemo', `Demo init failed: ${error.message}`);
});
