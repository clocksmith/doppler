import { log } from '../src/debug/index.js';
import { listPresets, createConverterConfig, detectPreset, resolvePreset } from '../src/config/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../src/config/schema/index.js';
import { listRegisteredModels, registerModel, removeRegisteredModel } from '../src/storage/registry.js';
import { formatBytes, getQuotaInfo } from '../src/storage/quota.js';
import { listStorageInventory, deleteStorageEntry } from '../src/storage/inventory.js';
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
import { loadRuntimePreset } from '../src/inference/browser-harness.js';
import { createPipeline } from '../src/inference/pipeline.js';
import { initDevice, getDevice, getKernelCapabilities, getPlatformConfig, isWebGPUAvailable } from '../src/gpu/device.js';
import { captureMemorySnapshot } from '../src/loader/memory-monitor.js';
import { destroyBufferPool } from '../src/memory/buffer-pool.js';
import { DiagnosticsController } from './diagnostics-controller.js';

const state = {
  runtimeOverride: null,
  runtimeOverrideBase: null,
  runtimeOverrideLabel: null,
  diagnosticsRuntimeConfig: null,
  diagnosticsRuntimePresetId: null,
  diagnosticsSelections: {},
  lastDiagnosticsSuite: null,
  lastDiffusionRequest: null,
  lastEnergyRequest: null,
  lastReport: null,
  lastReportInfo: null,
  lastMetrics: null,
  lastInferenceStats: null,
  lastMemoryStats: null,
  activePipeline: null,
  activePipelineModelId: null,
  activeModelId: null,
  modelTypeCache: {},
  modeModelId: { run: null, diffusion: null, energy: null },
  runAbortController: null,
  runGenerating: false,
  runLoading: false,
  diffusionGenerating: false,
  diffusionLoading: false,
  energyGenerating: false,
  energyLoading: false,
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
  energyDemoId: null,
  energyVliw: null,
  energyVliwBundle: null,
  energyVliwMeta: null,
};

const VLIW_DATASETS = {
  'vliw-simd': {
    id: 'vliw-simd',
    label: 'VLIW SIMD schedule (full kernel)',
    path: 'data/vliw-simd.json',
  },
};

const energyDatasetCache = new Map();

const ENERGY_DEMOS = [
  {
    id: 'quintel-cross',
    problem: 'quintel',
    label: 'Quintel: Cross (mirror + count)',
    description: 'Mirror X/Y with a count target. Produces a symmetric cross pattern.',
    defaults: {
      size: 5,
      displayThreshold: 0.2,
      countTarget: 12,
      rules: {
        mirrorX: true,
        mirrorY: true,
        diagonal: false,
        count: true,
      },
      weights: {
        symmetry: 1.0,
        count: 0.2,
        binarize: 0.01,
      },
      init: {
        mode: 'uniform',
        seed: 1337,
        scale: 0.6,
      },
      loop: {
        steps: 200,
        stepSize: 0.02,
        gradientScale: 0.5,
        convergence: 0.00001,
      },
    },
  },
  {
    id: 'quintel-diagonal',
    problem: 'quintel',
    label: 'Quintel: Diagonal symmetry',
    description: 'Diagonal symmetry with a softer count target.',
    defaults: {
      size: 5,
      displayThreshold: 0.35,
      countTarget: 10,
      rules: {
        mirrorX: false,
        mirrorY: false,
        diagonal: true,
        count: true,
      },
      weights: {
        symmetry: 0.8,
        count: 0.15,
        binarize: 0.02,
      },
      init: {
        mode: 'uniform',
        seed: 1337,
        scale: 0.55,
      },
      loop: {
        steps: 240,
        stepSize: 0.02,
        gradientScale: 0.5,
        convergence: 0.00001,
      },
    },
  },
  {
    id: 'quintel-symmetry',
    problem: 'quintel',
    label: 'Quintel: Symmetry only',
    description: 'Mirror constraints only (no count rule).',
    defaults: {
      size: 5,
      displayThreshold: 0.45,
      countTarget: 12,
      rules: {
        mirrorX: true,
        mirrorY: true,
        diagonal: false,
        count: false,
      },
      weights: {
        symmetry: 1.0,
        count: 0.0,
        binarize: 0.02,
      },
      init: {
        mode: 'uniform',
        seed: 1337,
        scale: 0.5,
      },
      loop: {
        steps: 160,
        stepSize: 0.02,
        gradientScale: 0.5,
        convergence: 0.00001,
      },
    },
  },
  {
    id: 'vliw-simd',
    problem: 'vliw',
    label: 'VLIW SIMD: Schedule search',
    description: 'Searches for a shorter VLIW SIMD schedule under slot caps (full kernel by default).',
    defaults: {
      displayThreshold: 0.5,
      vliw: {
        dataset: 'vliw-simd',
        bundleLimit: 0,
        restarts: 6,
        temperatureStart: 3.0,
        temperatureDecay: 0.99,
        mutationCount: 8,
      },
      init: {
        mode: 'normal',
        seed: 1337,
        scale: 0.35,
      },
      loop: {
        steps: 600,
        stepSize: 0.15,
        gradientScale: 1.0,
        convergence: 0,
      },
    },
  },
];

const DEFAULT_ENERGY_DEMO_ID = ENERGY_DEMOS[0]?.id || 'quintel-cross';

const ENERGY_METRIC_LABELS = {
  quintel: {
    symmetry: 'Symmetry',
    count: 'Count',
    binarize: 'Binarize',
  },
  vliw: {
    symmetry: 'Cycles',
    count: 'Utilization',
    binarize: 'Violations',
  },
};

const RUNTIME_PRESET_REGISTRY = [
  { id: '', label: 'none', base: false, override: true },
  { id: 'modes/debug', label: 'modes/debug', base: true, override: false },
  { id: 'modes/bench', label: 'modes/bench', base: true, override: false },
  { id: 'modes/production', label: 'modes/production', base: false, override: true },
  { id: 'modes/low-memory', label: 'modes/low-memory', base: false, override: true },
  { id: 'modes/simulation', label: 'modes/simulation', base: false, override: true },
  { id: 'modes/trace-layers', label: 'modes/trace-layers', base: false, override: true },
  { id: 'kernels/safe-q4k', label: 'kernels/safe-q4k', base: false, override: true },
  { id: 'kernels/fused-q4k', label: 'kernels/fused-q4k', base: false, override: true },
  { id: 'kernels/dequant-f16-q4k', label: 'kernels/dequant-f16-q4k', base: false, override: true },
  { id: 'kernels/dequant-f32-q4k', label: 'kernels/dequant-f32-q4k', base: false, override: true },
  { id: 'compute/f16-activations', label: 'compute/f16-activations', base: false, override: true },
  { id: 'compute/f16-batched', label: 'compute/f16-batched', base: false, override: true },
  { id: 'platform/metal-apple-q4k', label: 'platform/metal-apple-q4k', base: false, override: true },
  { id: 'model/gemma3-layer-probe', label: 'model/gemma3-layer-probe', base: false, override: true },
  { id: 'model/gemma2-pipeline', label: 'model/gemma2-pipeline', base: false, override: true },
  { id: 'model/gemma2-pipeline-debug', label: 'model/gemma2-pipeline-debug', base: false, override: true },
  { id: 'experiments/gemma3-verify', label: 'experiments/gemma3-verify', base: false, override: true },
  { id: 'experiments/gemma3-debug-q4k', label: 'experiments/gemma3-debug-q4k', base: false, override: true },
];

const DIAGNOSTICS_SUITE_INFO = {
  kernels: {
    description: 'Validates GPU kernels only (no model required).',
    requiresModel: false,
    requiresBenchIntent: false,
  },
  inference: {
    description: 'Runs a short generation with the Active model.',
    requiresModel: true,
    requiresBenchIntent: false,
  },
  bench: {
    description: 'Benchmarks tokens/sec for the Active model.',
    requiresModel: true,
    requiresBenchIntent: true,
  },
  debug: {
    description: 'Runs inference with debug tracing enabled by runtime config.',
    requiresModel: true,
    requiresBenchIntent: false,
  },
  diffusion: {
    description: 'Benchmarks diffusion generation using the Active model.',
    requiresModel: true,
    requiresBenchIntent: true,
  },
  energy: {
    description: 'Runs an energy loop with the Active model and reports convergence stats.',
    requiresModel: true,
    requiresBenchIntent: false,
  },
};

const BENCH_INTENTS = new Set(['investigate', 'calibrate']);
const DEFAULT_RUNTIME_PRESET = 'modes/debug';
const DIAGNOSTICS_DEFAULTS = {
  run: { suite: 'inference' },
  diffusion: { suite: 'diffusion' },
  energy: { suite: 'energy' },
  diagnostics: { suite: 'inference' },
};

const controller = new DiagnosticsController({ log });

function $(id) {
  return document.getElementById(id);
}

function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

function cloneRuntimeConfig(config) {
  try {
    return structuredClone(config);
  } catch {
    return JSON.parse(JSON.stringify(config));
  }
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
  if (state.runLoading) {
    setStatusIndicator('Loading model', 'info');
    return;
  }
  if (state.diffusionLoading) {
    setStatusIndicator('Loading diffusion', 'info');
    return;
  }
  if (state.energyLoading) {
    setStatusIndicator('Loading energy', 'info');
    return;
  }
  if (state.convertActive) {
    setStatusIndicator('Converting', 'info');
    return;
  }
  if (state.runGenerating) {
    setStatusIndicator('Generating', 'info');
    return;
  }
  if (state.diffusionGenerating) {
    setStatusIndicator('Generating', 'info');
    return;
  }
  if (state.energyGenerating) {
    setStatusIndicator('Running energy', 'info');
    return;
  }
  if (state.downloadActive) {
    setStatusIndicator('Downloading', 'info');
    return;
  }
  setStatusIndicator('Ready', 'success');
}

function getStatsMode() {
  if (state.uiMode === 'energy') return 'energy';
  if (state.uiMode === 'diffusion') return 'diffusion';
  if (state.uiMode === 'diagnostics' && state.lastDiagnosticsSuite === 'energy') {
    return 'energy';
  }
  if (state.uiMode === 'diagnostics' && state.lastDiagnosticsSuite === 'diffusion') {
    return 'diffusion';
  }
  const pipelineType = normalizeModelType(state.activePipeline?.manifest?.modelType);
  if (pipelineType === 'energy') return 'energy';
  if (pipelineType === 'diffusion') return 'diffusion';
  return 'text';
}

function setStatLabels(labels) {
  setText($('stat-tps-label'), labels.tps);
  setText($('stat-ttft-label'), labels.ttft);
  setText($('stat-prefill-label'), labels.prefill);
  setText($('stat-e2e-label'), labels.e2e);
  setText($('stat-decode-label'), labels.decode);
  setText($('stat-tokens-label'), labels.tokens);
}

function setRunLogLabels(labels) {
  setText($('run-log-ttft-label'), labels.ttft);
  setText($('run-log-prefill-label'), labels.prefill);
  setText($('run-log-decode-label'), labels.decode);
  setText($('run-log-e2e-label'), labels.e2e);
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
  app.dataset.mode = mode;
  document.querySelectorAll('.mode-tab').forEach((button) => {
    const isActive = button.dataset.mode === mode;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
  applyModeVisibility(mode);
  updatePerformancePanel();
  renderRunLog();
  if (mode === 'models') {
    refreshStorageInspector();
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

function formatScalar(value, digits = 4) {
  if (!Number.isFinite(value)) return '--';
  return value.toFixed(digits);
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

function getDiagnosticsDefaultSuite(mode) {
  return DIAGNOSTICS_DEFAULTS[mode]?.suite || 'inference';
}

function getDiagnosticsRuntimeConfig() {
  return state.diagnosticsRuntimeConfig || getRuntimeConfig();
}

async function refreshDiagnosticsRuntimeConfig(presetId) {
  const targetPreset = presetId || DEFAULT_RUNTIME_PRESET;
  const { runtime } = await loadRuntimePreset(targetPreset);
  const mergedOverride = getMergedRuntimeOverride();
  const mergedRuntime = mergedOverride ? mergeRuntimeOverrides(runtime, mergedOverride) : runtime;
  state.diagnosticsRuntimeConfig = mergedRuntime;
  state.diagnosticsRuntimePresetId = targetPreset;
  return mergedRuntime;
}

async function syncDiagnosticsDefaultsForMode(mode) {
  if (mode !== 'run' && mode !== 'diffusion' && mode !== 'energy' && mode !== 'diagnostics') return;
  const suiteSelect = $('diagnostics-suite');
  const presetSelect = $('runtime-preset');
  const selections = state.diagnosticsSelections[mode] || {};
  const targetSuite = selections.suite || getDiagnosticsDefaultSuite(mode);
  if (suiteSelect && targetSuite) {
    suiteSelect.value = targetSuite;
  }
  if (presetSelect) {
    const targetPreset = selections.preset || DEFAULT_RUNTIME_PRESET;
    presetSelect.value = targetPreset;
  }
  await applySelectedRuntimePreset();
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

function clearDiagnosticsOutput() {
  const container = $('diagnostics-output');
  const textEl = $('diagnostics-output-text');
  const canvas = $('diagnostics-output-canvas');
  if (textEl) textEl.textContent = '';
  if (canvas) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    canvas.hidden = true;
  }
  if (container) container.hidden = true;
}

function drawDiagnosticsCanvas(output) {
  const canvas = $('diagnostics-output-canvas');
  if (!canvas || !output) return;
  canvas.width = output.width;
  canvas.height = output.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const imageData = new ImageData(output.pixels, output.width, output.height);
  ctx.putImageData(imageData, 0, 0);
  canvas.hidden = false;
}

function renderDiagnosticsOutput(result, suite, captureOutput) {
  const container = $('diagnostics-output');
  if (!container) return;
  if (!captureOutput) {
    clearDiagnosticsOutput();
    return;
  }
  container.hidden = false;
  const textEl = $('diagnostics-output-text');
  const output = result?.output ?? null;
  if (suite === 'diffusion') {
    if (output && typeof output === 'object' && output.pixels) {
      if (textEl) textEl.textContent = '';
      drawDiagnosticsCanvas(output);
      return;
    }
    if (textEl) textEl.textContent = 'No diffusion output captured.';
    return;
  }
  if (suite === 'inference' || suite === 'debug') {
    if (typeof output === 'string' && output.length > 0) {
      if (textEl) textEl.textContent = output;
      return;
    }
    if (textEl) textEl.textContent = 'No output captured.';
    return;
  }
  if (textEl) textEl.textContent = 'Output is only captured for debug runs.';
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
  if (mode === 'energy') {
    return normalized === 'energy';
  }
  if (mode === 'run') {
    return normalized !== 'diffusion' && normalized !== 'energy';
  }
  return true;
}

function isModeModelSelectable(mode) {
  return mode === 'run' || mode === 'diffusion' || mode === 'energy';
}

function getModeModelLabel(mode) {
  if (mode === 'diffusion') return 'diffusion';
  if (mode === 'energy') return 'energy';
  if (mode === 'run') return 'text';
  return 'local';
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

function getDiagnosticsSuiteInfo(suite) {
  const key = String(suite || 'inference').trim().toLowerCase();
  return DIAGNOSTICS_SUITE_INFO[key] || DIAGNOSTICS_SUITE_INFO.inference;
}

function updateDiagnosticsGuidance() {
  const suiteSelect = $('diagnostics-suite');
  const modelSelect = $('diagnostics-model');
  const intentEl = $('diagnostics-intent');
  const suiteHelp = $('diagnostics-suite-help');
  const requirements = $('diagnostics-requirements');
  const runBtn = $('diagnostics-run-btn');
  const verifyBtn = $('diagnostics-verify-btn');
  if (!suiteSelect || !intentEl || !suiteHelp || !requirements) return;

  const suite = suiteSelect.value || getDiagnosticsDefaultSuite(state.uiMode);
  const info = getDiagnosticsSuiteInfo(suite);
  const runtimeConfig = getDiagnosticsRuntimeConfig();
  const intent = runtimeConfig?.shared?.tooling?.intent ?? null;
  const modelId = modelSelect?.value || '';

  intentEl.textContent = intent || 'unset';
  suiteHelp.textContent = info.description;

  const issues = [];
  if (!intent) {
    issues.push('Set runtime.shared.tooling.intent via preset or override.');
  } else if (info.requiresBenchIntent && !BENCH_INTENTS.has(intent)) {
    issues.push('Bench requires intent investigate or calibrate.');
  }
  if (info.requiresModel && !modelId) {
    issues.push('Select an Active model to run this suite.');
  }

  if (issues.length > 0) {
    requirements.textContent = issues.join(' ');
  } else {
    const ready = info.requiresModel ? `Ready. Using model ${modelId}.` : 'Ready. No model required.';
    requirements.textContent = ready;
  }

  const canVerify = Boolean(intent) && (!info.requiresBenchIntent || BENCH_INTENTS.has(intent));
  const canRun = canVerify && (!info.requiresModel || Boolean(modelId));
  if (verifyBtn) verifyBtn.disabled = !canVerify;
  if (runBtn) runBtn.disabled = !canRun;
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
  updateDiagnosticsGuidance();
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

function getRegistryModelId(entry) {
  if (!entry) return '';
  return entry.modelId || entry.id || '';
}

function getRegistryTags(entry) {
  if (!entry) return [];
  const tags = [];
  const quant = entry.quantization ? String(entry.quantization).toLowerCase() : '';
  if (quant) {
    if (quant === 'f16' || quant === 'bf16' || quant === 'f32') {
      tags.push(`weights:${quant}`);
    } else if (
      quant.startsWith('w') &&
      (quant.endsWith('f16') || quant.endsWith('bf16') || quant.endsWith('f32'))
    ) {
      tags.push(`weights:${quant.slice(1)}`);
    } else {
      tags.push(`quant:${quant}`);
    }
  }
  if (entry.hashAlgorithm) {
    tags.push(String(entry.hashAlgorithm));
  }
  return tags;
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
  const backendLabel = entry.backend ? entry.backend.toUpperCase() : 'storage';
  const confirmed = window.confirm(`Delete ${entry.modelId} (${sizeLabel}) from ${backendLabel}?`);
  if (!confirmed) return;

  try {
    await deleteStorageEntry(entry);
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

  let registryEntries = [];
  const registryIds = new Set();
  const registryById = new Map();
  try {
    registryEntries = await listRegisteredModels();
    for (const entry of registryEntries) {
      const id = getRegistryModelId(entry);
      if (!id) continue;
      registryIds.add(id);
      registryById.set(id, entry);
    }
  } catch (error) {
    log.warn('DopplerDemo', `Registry unavailable: ${error.message}`);
  }

  try {
    const inventory = await listStorageInventory();
    const storageEntries = inventory.entries.map((entry) => ({
      ...entry,
      registered: registryIds.has(entry.modelId),
      registryEntry: registryById.get(entry.modelId) || null,
    }));
    const storageIds = new Set(storageEntries.map((entry) => entry.modelId));
    const registryOnlyEntries = [];
    for (const [modelId, registryEntry] of registryById.entries()) {
      if (storageIds.has(modelId)) continue;
      const totalBytes = Number.isFinite(registryEntry?.totalSize)
        ? registryEntry.totalSize
        : Number.NaN;
      registryOnlyEntries.push({
        modelId,
        backend: registryEntry?.backend || 'unknown',
        root: '',
        totalBytes,
        fileCount: 0,
        shardCount: 0,
        hasManifest: false,
        registered: true,
        missingStorage: true,
        registryEntry,
      });
    }
    const entries = [...storageEntries, ...registryOnlyEntries];
    const systemEntries = inventory.systemEntries;
    const opfsRoots = inventory.opfsRoots;
    const backendParts = [];

    if (inventory.backendAvailability.opfs) {
      backendParts.push(
        opfsRoots.length ? `OPFS: ${opfsRoots.join(', ')}` : 'OPFS: empty'
      );
    } else {
      backendParts.push('OPFS: unavailable');
    }

    const idbName = runtime?.loading?.storage?.backend?.indexeddb?.dbName || 'indexeddb';
    if (inventory.backendAvailability.indexeddb) {
      backendParts.push(`IDB: ${idbName}`);
    } else {
      backendParts.push('IDB: unavailable');
    }
    backendEl.textContent = backendParts.join(' • ');

    entries.sort((a, b) => {
      const aMissing = a.missingStorage ? 1 : 0;
      const bMissing = b.missingStorage ? 1 : 0;
      if (aMissing !== bMissing) return aMissing - bMissing;
      const aSize = Number.isFinite(a.totalBytes) ? a.totalBytes : 0;
      const bSize = Number.isFinite(b.totalBytes) ? b.totalBytes : 0;
      return bSize - aSize;
    });

    const totals = new Map();
    const counts = new Map();
    for (const entry of storageEntries) {
      const backend = entry.backend || 'unknown';
      counts.set(backend, (counts.get(backend) || 0) + 1);
      if (Number.isFinite(entry.totalBytes)) {
        totals.set(backend, (totals.get(backend) || 0) + entry.totalBytes);
      }
    }

    const systemBytes = systemEntries.reduce((sum, entry) => sum + entry.totalBytes, 0);
    const summaryParts = [];
    if (entries.length) {
      summaryParts.push(`Models: ${entries.length}`);
    }
    if (registryOnlyEntries.length) {
      summaryParts.push(`Registry-only: ${registryOnlyEntries.length}`);
    }
    const orderedBackends = ['opfs', 'indexeddb', 'memory', 'unknown'];
    for (const backend of orderedBackends) {
      const count = counts.get(backend);
      if (!count) continue;
      const bytes = totals.get(backend);
      const label = backend === 'indexeddb' ? 'IDB' : backend.toUpperCase();
      if (Number.isFinite(bytes) && bytes > 0) {
        summaryParts.push(`${label}: ${formatBytes(bytes)} (${count})`);
      } else {
        summaryParts.push(`${label}: ${count}`);
      }
    }
    if (systemEntries.length) {
      summaryParts.push(`System: ${formatBytes(systemBytes)}`);
    }
    summaryEl.textContent = summaryParts.length
      ? summaryParts.join(' • ')
      : 'No models found';

    if (!entries.length) {
      listEl.innerHTML = '<div class="type-caption">No models found.</div>';
      setStorageInspectorStatus('Ready');
    } else {
      for (const entry of entries) {
        const row = document.createElement('div');
        row.className = 'storage-entry';
        if (entry.modelId === state.activeModelId) {
          row.classList.add('is-active');
        }
        if (entry.missingStorage) {
          row.classList.add('is-missing');
        }

        const main = document.createElement('div');
        main.className = 'storage-entry-main';

        const title = document.createElement('div');
        title.className = 'storage-entry-title';

        const name = document.createElement('span');
        name.className = 'type-caption';
        name.textContent = entry.modelId;
        title.appendChild(name);

        const backendTag = document.createElement('span');
        backendTag.className = 'storage-tag';
        backendTag.textContent = entry.backend === 'indexeddb' ? 'idb' : entry.backend;
        title.appendChild(backendTag);

        const tag = document.createElement('span');
        tag.className = `storage-tag${entry.registered ? '' : ' orphan'}`;
        tag.textContent = entry.registered ? 'registered' : 'orphan';
        title.appendChild(tag);

        const registryTags = getRegistryTags(entry.registryEntry);
        for (const registryTag of registryTags) {
          const registryTagEl = document.createElement('span');
          registryTagEl.className = 'storage-tag';
          registryTagEl.textContent = registryTag;
          title.appendChild(registryTagEl);
        }

        if (entry.missingStorage) {
          const missingStorage = document.createElement('span');
          missingStorage.className = 'storage-tag missing';
          missingStorage.textContent = 'registry only';
          title.appendChild(missingStorage);
        }

        if (!entry.hasManifest) {
          const missing = document.createElement('span');
          missing.className = 'storage-tag missing';
          missing.textContent = 'no manifest';
          title.appendChild(missing);
        }

        main.appendChild(title);

        const shardLabel = entry.missingStorage
          ? 'registry only'
          : entry.shardCount
            ? `${entry.shardCount} shards`
            : `${entry.fileCount} files`;
        const detail = document.createElement('span');
        detail.className = 'type-caption';
        const rootLabel = entry.root ? ` • root: ${entry.root}` : '';
        const sizeBytes = Number.isFinite(entry.totalBytes)
          ? entry.totalBytes
          : Number.isFinite(entry.registryEntry?.totalSize)
            ? entry.registryEntry.totalSize
            : Number.NaN;
        const sizeLabel = Number.isFinite(sizeBytes)
          ? formatBytes(sizeBytes)
          : 'unknown size';
        detail.textContent = entry.missingStorage
          ? `${sizeLabel} • ${shardLabel}`
          : `${sizeLabel} • ${shardLabel}${rootLabel}`;
        main.appendChild(detail);

        row.appendChild(main);

        if (!entry.missingStorage) {
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
          row.appendChild(actions);
          row.addEventListener('click', () => selectDiagnosticsModel(entry.modelId));
        }
        listEl.appendChild(row);
      }
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
        detail.className = 'type-caption';
        const systemRoot = entry.root ? ` • root: ${entry.root}` : '';
        detail.textContent = `${formatBytes(entry.totalBytes)} • ${entry.fileCount} files${systemRoot}`;
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
  const ttftEl = $('stat-ttft');
  const e2eEl = $('stat-e2e');
  const prefillEl = $('stat-prefill');
  const decodeEl = $('stat-decode');
  const tokensEl = $('stat-tokens');
  const mode = getStatsMode();

  const metrics = state.lastMetrics || {};
  const liveTps = state.runGenerating ? metrics.liveTokensPerSec : null;
  const tps = Number.isFinite(liveTps)
    ? liveTps
    : (Number.isFinite(metrics.tokensPerSec)
      ? metrics.tokensPerSec
      : (Number.isFinite(metrics.medianTokensPerSec) ? metrics.medianTokensPerSec : null));
  const stats = state.lastInferenceStats || {};
  if (mode === 'energy') {
    setStatLabels({
      tps: 'Steps/sec',
      ttft: 'Steps',
      prefill: 'Avg step',
      e2e: 'Energy',
      decode: 'Total',
      tokens: 'Shape',
    });

    const steps = Number.isFinite(stats.steps) ? stats.steps : null;
    const totalMs = Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : null;
    const energy = Number.isFinite(stats.energy) ? stats.energy : null;
    const avgStepMs = (steps != null && totalMs && totalMs > 0) ? totalMs / steps : null;
    const stepsPerSec = (steps != null && totalMs && totalMs > 0)
      ? steps / (totalMs / 1000)
      : null;

    setText(tpsEl, stepsPerSec != null ? stepsPerSec.toFixed(2) : '--');
    setText(ttftEl, steps != null ? String(steps) : '--');
    setText(prefillEl, avgStepMs != null ? formatMs(avgStepMs) : '--');
    setText(decodeEl, formatMs(totalMs));
    setText(e2eEl, energy != null ? formatScalar(energy, 6) : '--');

    const request = state.lastEnergyRequest || {};
    const shape = Array.isArray(request.shape) ? request.shape : null;
    if (shape && shape.length) {
      setText(tokensEl, shape.join(' x '));
    } else if (
      Number.isFinite(request.height) &&
      Number.isFinite(request.width) &&
      Number.isFinite(request.channels)
    ) {
      setText(tokensEl, `${request.height}x${request.width}x${request.channels}`);
    } else {
      setText(tokensEl, '--');
    }
    return;
  }
  if (mode === 'diffusion') {
    setStatLabels({
      tps: 'Steps/sec',
      ttft: 'Prompt',
      prefill: 'Denoise',
      e2e: 'Total',
      decode: 'VAE',
      tokens: 'Resolution / Steps',
    });

    const steps = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
    const denoiseMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
    const promptMs = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null;
    const vaeMs = Number.isFinite(stats.vaeTimeMs) ? stats.vaeTimeMs : null;
    const totalMs = Number.isFinite(stats.totalTimeMs)
      ? stats.totalTimeMs
      : (Number.isFinite(promptMs) && Number.isFinite(denoiseMs)
        ? promptMs + denoiseMs + (Number.isFinite(vaeMs) ? vaeMs : 0)
        : null);
    const stepsPerSec = (steps != null && denoiseMs && denoiseMs > 0)
      ? steps / (denoiseMs / 1000)
      : null;

    setText(tpsEl, stepsPerSec != null ? stepsPerSec.toFixed(2) : '--');
    setText(ttftEl, formatMs(promptMs));
    setText(prefillEl, formatMs(denoiseMs));
    setText(decodeEl, formatMs(vaeMs));
    setText(e2eEl, formatMs(totalMs));

    const request = state.lastDiffusionRequest || {};
    const width = Number.isFinite(request.width) ? request.width : null;
    const height = Number.isFinite(request.height) ? request.height : null;
    const stepsLabel = steps != null ? steps : '--';
    if (width && height) {
      setText(tokensEl, `${width}x${height} / ${stepsLabel}`);
    } else if (steps != null) {
      setText(tokensEl, `-- / ${stepsLabel}`);
    } else {
      setText(tokensEl, '--');
    }
    return;
  }

  setStatLabels({
    tps: 'Tokens/sec',
    ttft: 'TTFT',
    prefill: 'Prefill',
    e2e: 'End-to-end',
    decode: 'Decode',
    tokens: 'Prompt / Gen',
  });

  setText(tpsEl, tps !== null ? `${tps.toFixed(2)}` : '--');

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
  if (ttftEl) {
    setText(ttftEl, formatMs(ttftMs));
  }
  if (e2eEl) {
    setText(e2eEl, formatRate(e2eRate));
  }
  if (prefillEl) {
    if (prefillTokens == null && ttftMs == null && prefillRate == null) {
      setText(prefillEl, '--');
    } else {
      const tokenLabel = prefillTokens != null ? `${prefillTokens} tok` : '--';
      const rateLabel = prefillRate != null ? `${prefillRate.toFixed(2)} tok/s` : '--';
      setText(prefillEl, `${tokenLabel} @ ${rateLabel}`);
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

  if (tokensEl) {
    if (prefillTokens == null && decodeTokens == null) {
      setText(tokensEl, '--');
    } else {
      const promptLabel = prefillTokens != null ? prefillTokens : '--';
      const genLabel = decodeTokens != null ? decodeTokens : '--';
      setText(tokensEl, `${promptLabel} / ${genLabel}`);
    }
  }

}

function updateMemoryPanel(snapshot) {
  const poolStats = state.lastMemoryStats?.pool || null;
  const gpuStats = snapshot?.gpu || null;
  const gpuCurrent = Number.isFinite(gpuStats?.currentBytes) ? gpuStats.currentBytes : null;
  const gpuPeak = Number.isFinite(gpuStats?.peakBytes) ? gpuStats.peakBytes : null;
  const gpuRequested = Number.isFinite(gpuStats?.currentBytesRequested) ? gpuStats.currentBytesRequested : null;
  const activeBuffers = gpuStats?.activeBuffers ?? null;
  const pooledBuffers = gpuStats?.pooledBuffers ?? null;
  const gpuLimit = state.gpuMaxBytes || 0;

  setText($('stat-gpu-tracked'), Number.isFinite(gpuCurrent) ? formatBytes(gpuCurrent) : '--');
  setText($('stat-gpu-peak'), Number.isFinite(gpuPeak) ? formatBytes(gpuPeak) : '--');
  if (Number.isFinite(activeBuffers) && Number.isFinite(pooledBuffers)) {
    setText($('stat-gpu-buffers'), `${activeBuffers}/${pooledBuffers}`);
  } else {
    setText($('stat-gpu-buffers'), '--');
  }
  if (Number.isFinite(gpuRequested) && Number.isFinite(gpuCurrent)) {
    const requestedLabel = `${formatBytes(gpuRequested)} / ${formatBytes(gpuCurrent)}`;
    setText($('stat-gpu-requested'), requestedLabel);
  } else {
    setText($('stat-gpu-requested'), '--');
  }
  if (poolStats?.hitRate) {
    setText($('stat-gpu-hit'), poolStats.hitRate);
  } else {
    setText($('stat-gpu-hit'), '--');
  }
  if (gpuLimit) {
    setText($('stat-gpu-limit'), formatBytes(gpuLimit));
  } else {
    setText($('stat-gpu-limit'), '--');
  }

  const labelList = $('gpu-label-list');
  if (labelList) {
    const pool = state.activePipeline?.getBufferPool?.();
    const labelStats = typeof pool?.getLabelStats === 'function' ? pool.getLabelStats() : null;
    labelList.innerHTML = '';
    if (!labelStats || labelStats.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'type-caption';
      empty.textContent = 'No tracked buffers yet.';
      labelList.appendChild(empty);
    } else {
      const sorted = [...labelStats].sort((a, b) => (b.bytes || 0) - (a.bytes || 0));
      const top = sorted.slice(0, 6);
      for (const entry of top) {
        const row = document.createElement('div');
        row.className = 'stats-breakdown-row';

        const label = document.createElement('span');
        label.className = 'stats-breakdown-label';
        label.textContent = entry.label || 'unlabeled';

        const bytes = document.createElement('span');
        bytes.className = 'stats-breakdown-meta';
        bytes.textContent = Number.isFinite(entry.bytes) ? formatBytes(entry.bytes) : '--';

        const count = document.createElement('span');
        count.className = 'stats-breakdown-meta';
        count.textContent = Number.isFinite(entry.count) ? `${entry.count}` : '--';

        row.appendChild(label);
        row.appendChild(bytes);
        row.appendChild(count);
        labelList.appendChild(row);
      }
    }
  }

  const kvStats = state.lastMemoryStats?.kvCache || null;
  const kvAllocated = Number.isFinite(kvStats?.allocated) ? kvStats.allocated : null;
  const kvUsed = Number.isFinite(kvStats?.used) ? kvStats.used : null;
  const kvEff = Number.isFinite(kvStats?.efficiency) ? kvStats.efficiency : null;
  const kvSeq = Number.isFinite(kvStats?.seqLen) ? kvStats.seqLen : null;
  const kvMax = Number.isFinite(kvStats?.maxSeqLen) ? kvStats.maxSeqLen : null;
  const kvLayout = kvStats?.layout || null;

  setText($('stat-kv-allocated'), Number.isFinite(kvAllocated) ? formatBytes(kvAllocated) : '--');
  setText($('stat-kv-used'), Number.isFinite(kvUsed) ? formatBytes(kvUsed) : '--');
  if (Number.isFinite(kvEff)) {
    setText($('stat-kv-eff'), `${(kvEff * 100).toFixed(1)}%`);
  } else {
    setText($('stat-kv-eff'), '--');
  }
  if (Number.isFinite(kvSeq) && Number.isFinite(kvMax)) {
    setText($('stat-kv-seq'), `${kvSeq} / ${kvMax}`);
  } else {
    setText($('stat-kv-seq'), '--');
  }
  setText($('stat-kv-layout'), kvLayout || '--');

  const jsHeapUsed = Number.isFinite(snapshot?.jsHeapUsed) ? snapshot.jsHeapUsed : null;
  const jsHeapLimit = Number.isFinite(snapshot?.jsHeapLimit) ? snapshot.jsHeapLimit : null;
  if (Number.isFinite(jsHeapUsed) && Number.isFinite(jsHeapLimit) && jsHeapLimit > 0) {
    setText($('stat-heap'), `${formatBytes(jsHeapUsed)} / ${formatBytes(jsHeapLimit)}`);
  } else if (Number.isFinite(jsHeapUsed)) {
    setText($('stat-heap'), formatBytes(jsHeapUsed));
  } else {
    setText($('stat-heap'), '--');
  }

  if (state.systemMemoryBytes) {
    setText($('stat-ram-est'), formatBytes(state.systemMemoryBytes));
  } else {
    setText($('stat-ram-est'), '--');
  }

  const storageUsage = state.storageUsageBytes || 0;
  const storageQuota = state.storageQuotaBytes || 0;
  if (storageQuota) {
    setText($('stat-opfs'), `${formatBytes(storageUsage)} / ${formatBytes(storageQuota)}`);
  } else {
    setText($('stat-opfs'), '--');
  }
  setText($('stat-active-model'), state.activeModelId || 'none');
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

function renderRunLog() {
  const container = $('run-log-rows');
  if (!container) return;
  const mode = getStatsMode();
  container.innerHTML = '';
  const entries = state.runLog.filter((entry) => entry.mode === mode);
  if (mode === 'diffusion') {
    setRunLogLabels({
      ttft: 'Prompt',
      prefill: 'Denoise',
      decode: 'VAE',
      e2e: 'Total',
    });
  } else if (mode === 'energy') {
    setRunLogLabels({
      ttft: 'Steps',
      prefill: 'Avg step',
      decode: 'Total',
      e2e: 'Energy',
    });
  } else {
    setRunLogLabels({
      ttft: 'TTFT',
      prefill: 'Prefill',
      decode: 'Decode',
      e2e: 'E2E',
    });
  }
  for (const entry of entries) {
    const row = document.createElement('div');
    row.className = 'run-log-row';
    let cells;
    if (mode === 'diffusion') {
      cells = [
        entry.label,
        formatMs(entry.promptMs),
        formatMs(entry.denoiseMs),
        formatMs(entry.vaeMs),
        formatMs(entry.totalMs),
      ];
    } else if (mode === 'energy') {
      cells = [
        entry.label,
        entry.steps != null ? String(entry.steps) : '--',
        formatMs(entry.avgStepMs),
        formatMs(entry.totalMs),
        entry.energy != null ? formatScalar(entry.energy, 6) : '--',
      ];
    } else {
      cells = [
        entry.label,
        formatMs(entry.ttftMs),
        formatRate(entry.prefillRate),
        formatRate(entry.decodeRate),
        formatRate(entry.e2eRate),
      ];
    }
    for (const value of cells) {
      const cell = document.createElement('span');
      cell.textContent = value;
      row.appendChild(cell);
    }
    container.appendChild(row);
  }
}

function recordRunLog(stats, label, modeOverride) {
  if (!stats) return;
  const inferredMode = Number.isFinite(stats.vaeTimeMs)
    ? 'diffusion'
    : (Number.isFinite(stats.energy) || Array.isArray(stats.energyHistory) ? 'energy' : 'text');
  const mode = modeOverride || inferredMode;
  const prefillTokens = Number.isFinite(stats.prefillTokens) ? stats.prefillTokens : null;
  const decodeTokens = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
  const prefillTime = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null;
  const decodeTime = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
  const vaeTime = Number.isFinite(stats.vaeTimeMs) ? stats.vaeTimeMs : null;
  const totalTime = Number.isFinite(stats.totalTimeMs)
    ? stats.totalTimeMs
    : ((prefillTime && decodeTime) ? prefillTime + decodeTime : null);
  let entry = null;
  if (mode === 'diffusion') {
    entry = {
      mode,
      label,
      promptMs: prefillTime,
      denoiseMs: decodeTime,
      vaeMs: vaeTime,
      totalMs: Number.isFinite(stats.totalTimeMs)
        ? stats.totalTimeMs
        : ((prefillTime && decodeTime)
          ? prefillTime + decodeTime + (Number.isFinite(vaeTime) ? vaeTime : 0)
          : null),
    };
  } else if (mode === 'energy') {
    const steps = Number.isFinite(stats.steps) ? stats.steps : null;
    const energy = Number.isFinite(stats.energy) ? stats.energy : null;
    const totalMs = Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : null;
    const avgStepMs = (steps != null && totalMs && totalMs > 0) ? totalMs / steps : null;
    entry = {
      mode,
      label,
      steps,
      avgStepMs,
      totalMs,
      energy,
    };
  } else {
    entry = {
      mode,
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
  }
  state.runLog.unshift(entry);
  state.runLog = state.runLog.slice(0, 8);
  renderRunLog();
}

function readOptionalNumber(el, { integer = false } = {}) {
  const raw = el?.value;
  if (raw === '' || raw == null) return undefined;
  const parsed = integer ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function formatAutoValue(value, { integer = false } = {}) {
  if (!Number.isFinite(value)) return '--';
  if (integer) return `${Math.round(value)}`;
  const rounded = Math.round(value * 1000) / 1000;
  return `${rounded}`;
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

function updateEnergyStatus(message) {
  const status = $('energy-output-status');
  if (!status) return;
  setText(status, message || 'Idle');
}

function clearEnergyChart() {
  const canvas = $('energy-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getEnergyDemoById(id) {
  return ENERGY_DEMOS.find((demo) => demo.id === id) || null;
}

function setEnergyMetricLabels(problem) {
  const labels = ENERGY_METRIC_LABELS[problem] || ENERGY_METRIC_LABELS.quintel;
  setText($('energy-stat-label-symmetry'), labels.symmetry);
  setText($('energy-stat-label-count'), labels.count);
  setText($('energy-stat-label-binarize'), labels.binarize);
}

function toggleEnergyProblemControls(problem) {
  const quintelControls = $('energy-quintel-controls');
  const vliwControls = $('energy-vliw-controls');
  const summary = $('energy-kernel-summary')?.parentElement || null;
  const bundle = $('energy-bundle-view')?.parentElement || null;
  if (quintelControls) {
    quintelControls.hidden = problem !== 'quintel';
  }
  if (vliwControls) {
    vliwControls.hidden = problem !== 'vliw';
  }
  if (summary) {
    summary.hidden = problem !== 'vliw';
  }
  if (bundle) {
    bundle.hidden = problem !== 'vliw';
  }
}

function syncEnergyDemoSelection() {
  const select = $('energy-demo-select');
  if (!select) return;
  const selected = select.value || state.energyDemoId || DEFAULT_ENERGY_DEMO_ID;
  const demo = getEnergyDemoById(selected) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
  if (!demo) return;
  state.energyDemoId = demo.id;
  if (select.value !== demo.id) {
    select.value = demo.id;
  }
  setText($('energy-demo-description'), demo.description || '');
  setEnergyMetricLabels(demo.problem || 'quintel');
  toggleEnergyProblemControls(demo.problem || 'quintel');
}

function populateEnergyDemoSelect() {
  const select = $('energy-demo-select');
  if (!select) return;
  select.innerHTML = '';
  ENERGY_DEMOS.forEach((demo) => {
    const option = document.createElement('option');
    option.value = demo.id;
    option.textContent = demo.label;
    select.appendChild(option);
  });
  const initial = state.energyDemoId || DEFAULT_ENERGY_DEMO_ID;
  const demo = getEnergyDemoById(initial) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
  if (!demo) return;
  state.energyDemoId = demo.id;
  select.value = demo.id;
  setText($('energy-demo-description'), demo.description || '');
  setEnergyMetricLabels(demo.problem || 'quintel');
  toggleEnergyProblemControls(demo.problem || 'quintel');
  applyEnergyDemoDefaults(demo);
}

function applyEnergyDemoDefaults(demo) {
  if (!demo || !demo.defaults) return;
  const defaults = demo.defaults;
  const energyQuintelSize = $('energy-quintel-size');
  const energyQuintelThreshold = $('energy-quintel-threshold');
  const energyQuintelCountTarget = $('energy-quintel-count-target');
  const energyRuleMirrorX = $('energy-rule-mirror-x');
  const energyRuleMirrorY = $('energy-rule-mirror-y');
  const energyRuleDiagonal = $('energy-rule-diagonal');
  const energyRuleCount = $('energy-rule-count');
  const energyWeightSymmetry = $('energy-weight-symmetry');
  const energyWeightCount = $('energy-weight-count');
  const energyWeightBinarize = $('energy-weight-binarize');
  const energyInitMode = $('energy-init-mode');
  const energyInitSeed = $('energy-init-seed');
  const energyInitScale = $('energy-init-scale');
  const energySteps = $('energy-steps');
  const energyStepSize = $('energy-step-size');
  const energyGradientScale = $('energy-gradient-scale');
  const energyConvergence = $('energy-convergence');
  const energyVliwDataset = $('energy-vliw-dataset');
  const energyVliwBundleLimit = $('energy-vliw-bundle-limit');
  const energyVliwRestarts = $('energy-vliw-restarts');
  const energyVliwTempStart = $('energy-vliw-temp-start');
  const energyVliwTempDecay = $('energy-vliw-temp-decay');
  const energyVliwMutation = $('energy-vliw-mutation');

  if (energyQuintelSize && Number.isFinite(defaults.size)) {
    energyQuintelSize.value = String(defaults.size);
  }
  if (energyQuintelThreshold && Number.isFinite(defaults.displayThreshold)) {
    energyQuintelThreshold.value = String(defaults.displayThreshold);
  }
  if (energyQuintelCountTarget && Number.isFinite(defaults.countTarget)) {
    energyQuintelCountTarget.value = String(defaults.countTarget);
  }
  if (energyRuleMirrorX && typeof defaults.rules?.mirrorX === 'boolean') {
    energyRuleMirrorX.checked = defaults.rules.mirrorX;
  }
  if (energyRuleMirrorY && typeof defaults.rules?.mirrorY === 'boolean') {
    energyRuleMirrorY.checked = defaults.rules.mirrorY;
  }
  if (energyRuleDiagonal && typeof defaults.rules?.diagonal === 'boolean') {
    energyRuleDiagonal.checked = defaults.rules.diagonal;
  }
  if (energyRuleCount && typeof defaults.rules?.count === 'boolean') {
    energyRuleCount.checked = defaults.rules.count;
  }
  if (energyWeightSymmetry && Number.isFinite(defaults.weights?.symmetry)) {
    energyWeightSymmetry.value = String(defaults.weights.symmetry);
  }
  if (energyWeightCount && Number.isFinite(defaults.weights?.count)) {
    energyWeightCount.value = String(defaults.weights.count);
  }
  if (energyWeightBinarize && Number.isFinite(defaults.weights?.binarize)) {
    energyWeightBinarize.value = String(defaults.weights.binarize);
  }
  if (energyInitMode && defaults.init?.mode) {
    energyInitMode.value = defaults.init.mode;
  }
  if (energyInitSeed && Number.isFinite(defaults.init?.seed)) {
    energyInitSeed.value = String(defaults.init.seed);
  }
  if (energyInitScale && Number.isFinite(defaults.init?.scale)) {
    energyInitScale.value = String(defaults.init.scale);
  }
  if (energySteps && Number.isFinite(defaults.loop?.steps)) {
    energySteps.value = String(defaults.loop.steps);
  }
  if (energyStepSize && Number.isFinite(defaults.loop?.stepSize)) {
    energyStepSize.value = String(defaults.loop.stepSize);
  }
  if (energyGradientScale && Number.isFinite(defaults.loop?.gradientScale)) {
    energyGradientScale.value = String(defaults.loop.gradientScale);
  }
  if (energyConvergence && Number.isFinite(defaults.loop?.convergence)) {
    energyConvergence.value = String(defaults.loop.convergence);
  }
  if (energyVliwDataset && defaults.vliw?.dataset) {
    energyVliwDataset.value = defaults.vliw.dataset;
  }
  if (energyVliwBundleLimit && Number.isFinite(defaults.vliw?.bundleLimit)) {
    energyVliwBundleLimit.value = String(defaults.vliw.bundleLimit);
  }
  if (energyVliwRestarts && Number.isFinite(defaults.vliw?.restarts)) {
    energyVliwRestarts.value = String(defaults.vliw.restarts);
  }
  if (energyVliwTempStart && Number.isFinite(defaults.vliw?.temperatureStart)) {
    energyVliwTempStart.value = String(defaults.vliw.temperatureStart);
  }
  if (energyVliwTempDecay && Number.isFinite(defaults.vliw?.temperatureDecay)) {
    energyVliwTempDecay.value = String(defaults.vliw.temperatureDecay);
  }
  if (energyVliwMutation && Number.isFinite(defaults.vliw?.mutationCount)) {
    energyVliwMutation.value = String(defaults.vliw.mutationCount);
  }
}

async function loadVliwDataset(datasetId) {
  const entry = VLIW_DATASETS[datasetId];
  if (!entry) {
    throw new Error(`Unknown VLIW dataset "${datasetId}".`);
  }
  if (energyDatasetCache.has(datasetId)) {
    return energyDatasetCache.get(datasetId);
  }
  const response = await fetch(entry.path);
  if (!response.ok) {
    throw new Error(`Failed to load VLIW dataset: ${response.status}`);
  }
  const payload = await response.json();
  energyDatasetCache.set(datasetId, payload);
  return payload;
}

function sliceVliwDataset(dataset, bundleLimit) {
  if (!dataset || !Array.isArray(dataset.tasks)) {
    return { tasks: [], caps: {} };
  }
  const rawLimit = Number.isFinite(bundleLimit) ? Math.floor(bundleLimit) : null;
  const limit = rawLimit && rawLimit > 0 ? Math.max(1, rawLimit) : null;
  const tasks = limit == null
    ? dataset.tasks
    : dataset.tasks.filter((task) => (task.bundle ?? 0) < limit);
  const idMap = new Map();
  const remapped = [];
  let maxBundle = -1;
  tasks.forEach((task, index) => {
    idMap.set(task.id, index);
    const bundle = Number.isFinite(task.bundle) ? task.bundle : 0;
    if (bundle > maxBundle) maxBundle = bundle;
    remapped.push({ ...task, id: index });
  });
  remapped.forEach((task) => {
    const deps = Array.isArray(task.deps) ? task.deps : [];
    task.deps = deps.map((dep) => idMap.get(dep)).filter((dep) => dep != null);
  });
  return {
    tasks: remapped,
    caps: dataset.caps ?? {},
    bundleCount: maxBundle + 1,
    taskCount: remapped.length,
  };
}

function clearEnergyBoard() {
  const board = $('energy-board');
  if (!board) return;
  board.innerHTML = '';
  clearEnergyVector();
  clearEnergyIntensityBoard();
  clearEnergyKernelSummary();
  clearEnergyBundleView();
}

function clearEnergyVector() {
  const vector = $('energy-vector');
  if (!vector) return;
  vector.textContent = '';
}

function clearEnergyIntensityBoard() {
  const board = $('energy-board-intensity');
  if (!board) return;
  board.innerHTML = '';
}

function clearEnergyKernelSummary() {
  const summary = $('energy-kernel-summary');
  if (!summary) return;
  summary.textContent = '';
}

function clearEnergyBundleView() {
  const view = $('energy-bundle-view');
  if (view) view.textContent = '';
  const select = $('energy-vliw-bundle-select');
  if (select) select.innerHTML = '';
}

function resolveEnergyGrid(shapeOrSize) {
  if (Array.isArray(shapeOrSize) && shapeOrSize.length >= 2) {
    const rows = Math.max(1, Math.floor(shapeOrSize[0]));
    const cols = Math.max(1, Math.floor(shapeOrSize[1]));
    return { rows, cols };
  }
  const size = Math.max(1, Math.floor(shapeOrSize ?? 1));
  return { rows: size, cols: size };
}

function renderEnergyBoard(state, shapeOrSize, threshold) {
  const board = $('energy-board');
  if (!board) return;
  board.innerHTML = '';
  if (!state) return;
  const { rows, cols } = resolveEnergyGrid(shapeOrSize);
  const safeThreshold = Number.isFinite(threshold) ? threshold : 0.5;
  board.style.setProperty('--energy-grid-size', `${cols}`);
  const cellCount = rows * cols;
  for (let i = 0; i < cellCount; i++) {
    const cell = document.createElement('div');
    cell.className = 'energy-cell';
    const value = state[i];
    if (Number.isFinite(value) && value >= safeThreshold) {
      cell.classList.add('is-on');
    }
    if (Number.isFinite(value)) {
      cell.title = value.toFixed(2);
    }
    board.appendChild(cell);
  }
  renderEnergyVector(state, rows, cols, safeThreshold);
  renderEnergyIntensityBoard(state, rows, cols);
}

function renderEnergyVector(state, rows, cols, threshold) {
  const vector = $('energy-vector');
  if (!vector) return;
  vector.textContent = '';
  if (!state || !Number.isFinite(rows) || !Number.isFinite(cols)) return;
  const gridRows = Math.max(1, Math.floor(rows));
  const gridCols = Math.max(1, Math.floor(cols));
  const safeThreshold = Number.isFinite(threshold) ? threshold : 0.5;
  const lines = [];
  for (let row = 0; row < gridRows; row++) {
    const cells = [];
    for (let col = 0; col < gridCols; col++) {
      const index = row * gridCols + col;
      const value = state[index];
      const bit = Number.isFinite(value) && value >= safeThreshold ? '1' : '0';
      cells.push(bit);
    }
    lines.push(cells.join(' '));
  }
  vector.textContent = lines.join('\n');
}

function renderEnergyIntensityBoard(state, rows, cols) {
  const board = $('energy-board-intensity');
  if (!board) return;
  board.innerHTML = '';
  if (!state || !Number.isFinite(rows) || !Number.isFinite(cols)) return;
  const gridRows = Math.max(1, Math.floor(rows));
  const gridCols = Math.max(1, Math.floor(cols));
  board.style.setProperty('--energy-grid-size', `${gridCols}`);
  const cellCount = gridRows * gridCols;
  for (let i = 0; i < cellCount; i++) {
    const cell = document.createElement('div');
    cell.className = 'energy-cell';
    const value = state[i];
    if (Number.isFinite(value)) {
      const alpha = Math.max(0, Math.min(1, value));
      cell.style.backgroundColor = `rgba(0, 0, 0, ${alpha.toFixed(3)})`;
      cell.title = value.toFixed(2);
    }
    board.appendChild(cell);
  }
}

function formatVliwSlotLabel(engine, slotIndex) {
  if (!engine) return `slot${slotIndex}`;
  return `${engine}${slotIndex}`;
}

function renderVliwKernelSummary(summary, datasetMeta) {
  const summaryEl = $('energy-kernel-summary');
  if (!summaryEl) return;
  if (!summary) {
    summaryEl.textContent = '';
    return;
  }
  const lines = [];
  if (datasetMeta?.label) {
    lines.push(`Dataset: ${datasetMeta.label}`);
  }
  if (Number.isFinite(datasetMeta?.bundleCount)) {
    lines.push(`Bundles: ${datasetMeta.bundleCount}`);
  }
  if (Number.isFinite(datasetMeta?.taskCount)) {
    lines.push(`Tasks: ${datasetMeta.taskCount}`);
  }
  if (Number.isFinite(datasetMeta?.baselineCycles)) {
    lines.push(`Baseline cycles: ${datasetMeta.baselineCycles}`);
  }
  if (Number.isFinite(summary.bestCycles)) {
    lines.push(`Best cycles: ${summary.bestCycles}`);
  }
  if (Number.isFinite(summary.utilization)) {
    lines.push(`Utilization: ${formatScalar(summary.utilization, 4)}`);
  }
  if (Array.isArray(summary.candidates) && summary.candidates.length) {
    lines.push('Top candidates:');
    summary.candidates.forEach((candidate, index) => {
      const parts = [
        `#${index + 1}`,
        `restart ${candidate.restart}`,
        `cycles ${candidate.cycles}`,
        `util ${formatScalar(candidate.utilization, 4)}`,
        `viol ${candidate.violations}`,
        `steps ${candidate.steps}`,
      ];
      lines.push(`  ${parts.join(' • ')}`);
    });
  }
  summaryEl.textContent = lines.join('\n');
}

function populateVliwBundleSelect(bundleCount) {
  const select = $('energy-vliw-bundle-select');
  if (!select) return;
  select.innerHTML = '';
  const allOption = document.createElement('option');
  allOption.value = '';
  allOption.textContent = 'All bundles';
  select.appendChild(allOption);
  if (!Number.isFinite(bundleCount) || bundleCount <= 0) return;
  for (let i = 0; i < bundleCount; i++) {
    const option = document.createElement('option');
    option.value = String(i);
    option.textContent = `Bundle ${i}`;
    select.appendChild(option);
  }
}

function renderVliwBundleView(vliwState, selectedBundle) {
  const view = $('energy-bundle-view');
  if (!view) return;
  view.textContent = '';
  if (!vliwState || !vliwState.schedule) return;
  const { slotAssignments, slotEngines, slotIndices } = vliwState.schedule;
  if (!slotAssignments || !slotEngines || !slotIndices) return;
  const slotsPerCycle = slotEngines.length;
  if (!slotsPerCycle) return;
  const cycles = Math.floor(slotAssignments.length / slotsPerCycle);
  const lines = [];
  const showBundle = Number.isFinite(selectedBundle) ? selectedBundle : null;
  for (let cycle = 0; cycle < cycles; cycle++) {
    const parts = [`C${String(cycle).padStart(4, '0')}`];
    const baseIndex = cycle * slotsPerCycle;
    for (let slot = 0; slot < slotsPerCycle; slot++) {
      const taskId = slotAssignments[baseIndex + slot];
      const engine = slotEngines[slot];
      const slotIndex = slotIndices[slot];
      if (taskId == null || taskId < 0) {
        parts.push(`${formatVliwSlotLabel(engine, slotIndex)}=--`);
        continue;
      }
      const meta = vliwState.taskMeta?.[taskId] || {};
      const bundle = Number.isFinite(meta.bundle) ? meta.bundle : '--';
      const deps = Number.isFinite(meta.deps) ? meta.deps : 0;
      const reads = Number.isFinite(meta.reads) ? meta.reads : 0;
      const writes = Number.isFinite(meta.writes) ? meta.writes : 0;
      const highlight = showBundle != null && bundle === showBundle;
      const prefix = highlight ? '*' : '';
      parts.push(
        `${prefix}${formatVliwSlotLabel(engine, slotIndex)}=${taskId}[b${bundle} d${deps} r${reads} w${writes}]`,
      );
    }
    lines.push(parts.join(' '));
  }
  view.textContent = lines.join('\n');
}

function drawEnergyChart(history = []) {
  const canvas = $('energy-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  clearEnergyChart();

  const values = Array.isArray(history)
    ? history.filter((value) => Number.isFinite(value))
    : [];
  if (!values.length) return;

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const range = maxValue - minValue || 1;
  const padding = 12;
  const width = canvas.width;
  const height = canvas.height;

  ctx.strokeStyle = '#111';
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const t = values.length > 1 ? index / (values.length - 1) : 0;
    const x = padding + t * (width - padding * 2);
    const y = height - padding - ((value - minValue) / range) * (height - padding * 2);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function updateEnergyStats(result) {
  if (!result) {
    setText($('energy-stat-steps'), '--');
    setText($('energy-stat-energy'), '--');
    setText($('energy-stat-symmetry'), '--');
    setText($('energy-stat-count'), '--');
    setText($('energy-stat-binarize'), '--');
    setText($('energy-stat-dtype'), '--');
    setText($('energy-stat-backend'), '--');
    setText($('energy-stat-shape'), '--');
    setText($('energy-stat-mean'), '--');
    setText($('energy-stat-std'), '--');
    return;
  }
  const problem = result.problem || 'quintel';
  setText($('energy-stat-steps'), Number.isFinite(result.steps) ? String(result.steps) : '--');
  setText($('energy-stat-energy'), Number.isFinite(result.energy) ? formatScalar(result.energy, 6) : '--');
  if (problem === 'vliw' && result.metrics) {
    setText($('energy-stat-symmetry'), formatScalar(result.metrics.cycles, 0));
    setText($('energy-stat-count'), formatScalar(result.metrics.utilization, 4));
    setText($('energy-stat-binarize'), formatScalar(result.metrics.violations, 0));
  } else {
    setText($('energy-stat-symmetry'), formatScalar(result.energyComponents?.symmetry, 6));
    setText($('energy-stat-count'), formatScalar(result.energyComponents?.count, 6));
    setText($('energy-stat-binarize'), formatScalar(result.energyComponents?.binarize, 6));
  }
  setText($('energy-stat-dtype'), result.dtype || '--');
  setText($('energy-stat-backend'), result.backend || '--');
  const shape = Array.isArray(result.shape) ? result.shape.join(' x ') : '--';
  setText($('energy-stat-shape'), shape);
  setText($('energy-stat-mean'), formatScalar(result.stateStats?.mean, 6));
  setText($('energy-stat-std'), formatScalar(result.stateStats?.std, 6));
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

  if (problem !== 'vliw') {
    state.energyVliw = null;
    state.energyVliwBundle = null;
    state.energyVliwMeta = null;
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
    const datasetId = $('energy-vliw-dataset')?.value || 'vliw-simd';
    const bundleLimit = readOptionalNumber($('energy-vliw-bundle-limit'), { integer: true });
    const restarts = readOptionalNumber($('energy-vliw-restarts'), { integer: true });
    const tempStart = readOptionalNumber($('energy-vliw-temp-start'));
    const tempDecay = readOptionalNumber($('energy-vliw-temp-decay'));
    const mutationCount = readOptionalNumber($('energy-vliw-mutation'), { integer: true });
    const dataset = await loadVliwDataset(datasetId);
    const sliced = sliceVliwDataset(dataset, bundleLimit);
    state.energyVliwMeta = {
      label: dataset.label || VLIW_DATASETS[datasetId]?.label || datasetId,
      bundleCount: sliced.bundleCount ?? dataset.bundleCount,
      taskCount: sliced.taskCount ?? sliced.tasks?.length ?? dataset.taskCount,
      baselineCycles: dataset.baselineCycles ?? dataset.bundleCount,
    };
    request.vliw = {
      tasks: sliced.tasks,
      caps: sliced.caps,
      search: {
        restarts,
        temperatureStart: tempStart,
        temperatureDecay: tempDecay,
        mutationCount,
      },
    };
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
    const result = await pipeline.generate(request);
    if (problem === 'vliw') {
      state.energyVliw = {
        schedule: result?.schedule || null,
        taskMeta: result?.taskMeta || null,
      };
      const candidates = Array.isArray(result?.candidates) ? result.candidates.slice() : [];
      candidates.sort((a, b) => a.cycles - b.cycles);
      const summary = {
        bestCycles: result?.metrics?.cycles ?? null,
        utilization: result?.metrics?.utilization ?? null,
        candidates: candidates.slice(0, 6),
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
  state.energyVliwBundle = null;
  state.energyVliwMeta = null;
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
  let downloadedModelId = modelIdOverride ?? null;
  updateDownloadStatus({ status: 'Starting...', percent: 0, downloadedBytes: 0, totalBytes: 0 });
  try {
    await downloadModel(baseUrl, (progress) => {
      if (!progress) return;
      state.activeDownloadId = progress.modelId || modelIdOverride || null;
      downloadedModelId = progress.modelId || downloadedModelId;
      updateDownloadStatus(progress);
    }, { modelId: modelIdOverride });
    if (downloadedModelId) {
      await registerDownloadedModel(downloadedModelId);
    }
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
  const presetLabel = presetId || DEFAULT_RUNTIME_PRESET;
  if (state.runtimeOverride) {
    const labels = [];
    if (state.runtimeOverrideLabel) {
      labels.push(state.runtimeOverrideLabel);
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
  return state.runtimeOverrideBase;
}

async function applySelectedRuntimePreset() {
  const presetSelect = $('runtime-preset');
  if (!presetSelect) return;
  const presetId = presetSelect.value || DEFAULT_RUNTIME_PRESET;
  if (!presetSelect.value) {
    presetSelect.value = presetId;
  }
  const mergedOverride = getMergedRuntimeOverride();
  state.runtimeOverride = mergedOverride;
  updateRuntimeConfigStatus(presetId);
  try {
    await refreshDiagnosticsRuntimeConfig(presetId);
    updateDiagnosticsGuidance();
  } catch (error) {
    updateDiagnosticsStatus(`Preset error: ${error.message}`, true);
  }
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
  const energyClear = $('energy-clear-btn');

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
    const mode = state.uiMode;
    if (mode === 'run' || mode === 'diffusion' || mode === 'energy') {
      state.diagnosticsSelections[mode] = {
        ...(state.diagnosticsSelections[mode] || {}),
        preset: runtimePreset.value || DEFAULT_RUNTIME_PRESET,
      };
    }
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
    if (mode === 'run' || mode === 'diffusion' || mode === 'energy') {
      state.diagnosticsSelections[mode] = {
        ...(state.diagnosticsSelections[mode] || {}),
        suite: diagnosticsSuite.value || getDiagnosticsDefaultSuite(mode),
      };
    }
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
    runtimeFile.value = '';
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

  const energyBundleSelect = $('energy-vliw-bundle-select');
  energyBundleSelect?.addEventListener('change', () => {
    const value = energyBundleSelect.value;
    const bundle = value === '' ? null : Number.parseInt(value, 10);
    state.energyVliwBundle = Number.isFinite(bundle) ? bundle : null;
    renderVliwBundleView(state.energyVliw, state.energyVliwBundle);
  });

  updateRunAutoLabels();
}

async function init() {
  setStatusIndicator('Initializing', 'info');
  bindUI();
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
