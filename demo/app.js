import {
  log,
  listPresets,
  createConverterConfig,
  detectPreset,
  resolvePreset,
  getRuntimeConfig,
  setRuntimeConfig,
  DEFAULT_MANIFEST_INFERENCE,
  formatBytes,
  listRegisteredModels,
  registerModel,
  openModelStore,
  writeShard,
  loadManifestFromStore,
  loadTensorsFromStore,
  saveManifest,
  saveTensorsToStore,
  saveTokenizer,
  saveTokenizerModel,
  saveAuxFile,
  loadTokenizerFromStore,
  loadTokenizerModelFromStore,
  parseManifest,
  getManifest,
  setManifest,
  clearManifest,
  classifyTensorRole,
  convertModel,
  createRemoteModelSources,
  isConversionSupported,
  buildManifestInference,
  inferEmbeddingOutputConfig,
  pickModelDirectory,
  pickModelFiles,
  createPipeline,
  initDevice,
  getDevice,
  getKernelCapabilities,
  getPlatformConfig,
  isWebGPUAvailable,
  captureMemorySnapshot,
  destroyBufferPool,
} from '@doppler/core';
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
  decodeDiagnosticsProfileId,
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
  startDownloadFromBaseUrl,
  pauseActiveDownload,
  resumeActiveDownload,
  cancelActiveDownload,
} from './app/downloads/index.js';

const controller = new DiagnosticsController({ log });

const PRIMARY_MODES = new Set(['run', 'embedding', 'diffusion', 'energy']);
let modelListRefreshVersion = 0;
const DEFAULT_MODEL_AVAILABILITY = Object.freeze({ total: 0, run: 0, embedding: 0, diffusion: 0, energy: 0 });
const QUICK_MODEL_CATALOG_URL = new URL('../models/catalog.json', import.meta.url).toString();
const RUN_STARTER_PROMPTS = Object.freeze([
  'is potential energy real?',
  'compare zig to rust in elvish',
  'eat your cake and have it too',
  'pivot to neurosymbolic reasoning',
  'write a poem about an elephant that is bullish on QQQ',
  'explain why a toddler is exactly like a neural network',
  'explain the difference between the star trek migratation and star wars trek',
  'prove termination for a recursive functional agent using lean four and inductive types',
  'describe a toy store where the shelves are sorted by cognitive development stages and every single game has a proof of educational value attached',
  'is human intuition just a fast, low-energy heuristic that our biological hardware runs when the cost of slow, symbolic reasoning is too high for survival',
  'write a technical fable about an agent tasked with solving a paradox, forever rolling a high-energy gradient up a hill only for it to reset at every epoch',
]);
const EMBEDDING_DEMO_DOCUMENT_CATALOG = Object.freeze([
  Object.freeze({
    id: 'doc_webgpu_local',
    title: 'Local-First WebGPU',
    text: 'Local-first AI apps run inference in the browser using WebGPU and store model shards in OPFS for offline performance.',
  }),
  Object.freeze({
    id: 'doc_formal_methods',
    title: 'Formal Methods',
    text: 'Lean proofs can verify termination and memory-safety properties for recursive systems code with clear inductive structure.',
  }),
  Object.freeze({
    id: 'doc_market_qqq',
    title: 'Market Commentary',
    text: 'QQQ reflects large-cap technology exposure; risk management depends on volatility, drawdown tolerance, and rebalance discipline.',
  }),
  Object.freeze({
    id: 'doc_kv_cache',
    title: 'KV Cache Behavior',
    text: 'Transformer decoding reuses key/value cache state; resetting context between runs prevents prompt leakage and keeps measurements independent.',
  }),
  Object.freeze({
    id: 'doc_pkg_delivery',
    title: 'Support Delivery Case',
    text: 'Customers reporting damaged packages need replacement workflows, photo evidence handling, and clear refund timelines in support tooling.',
  }),
  Object.freeze({
    id: 'doc_formal_agent',
    title: 'Verified Agents',
    text: 'Recursive agents can be modeled with inductive types, then proven terminating so orchestration loops do not run forever in production.',
  }),
  Object.freeze({
    id: 'doc_diffusion',
    title: 'Image Generation',
    text: 'Diffusion inference denoises latent tensors over multiple steps, then decodes through a VAE to produce a final image.',
  }),
  Object.freeze({
    id: 'doc_energy_model',
    title: 'Energy Optimization',
    text: 'Energy-based solvers iteratively minimize objective functions and can visualize convergence as energy drops over time.',
  }),
  Object.freeze({
    id: 'doc_data_governance',
    title: 'Data Governance',
    text: 'Local storage policies should track model provenance, hash integrity, and retention windows for reproducible deployments.',
  }),
]);
const EMBEDDING_DEMO_DOCUMENT_COUNT = 3;
const DIFFUSION_STARTER_PROMPTS = Object.freeze([
  'A photo-realistic architectural render of a boutique toy store in Williamsburg, Brooklyn; matte black metal frame, floor-to-ceiling glass, warm wooden shelves with minimalist board games and wooden toys, soft morning sidewalk light.',
  "A vector logo for a software project named Doppler, cyber-industrial and minimalist, deep charcoal and neon teal palette, 90s arcade energy meets modern developer tooling.",
  'A top-down cinematic shot of a disassembled Framework DIY laptop next to a custom mechanical keyboard with translucent keycaps and coiled cables, shallow bokeh, texture-rich PCB details.',
  'A digital artwork of an ouroboros made from glowing fiber-optic cables and circuit board traces, dark background, high contrast, precise luminous edges.',
  'A candid documentary-style photo of a museum visitor looking up at a massive dinosaur skeleton in the American Museum of Natural History, soft natural light, slight desaturation.',
  'A 2022 Audi Q3 with honeycomb mesh grille and low-profile roof racks parked on a cobblestone street in DUMBO, cinematic automotive lighting, crisp reflections.',
  'A macro shot of a complex tabletop strategy game in progress, wooden pieces, intricate cards, polyhedral dice on dark walnut, warm cozy lighting.',
  'A macro, high-contrast black-and-white photo of a Somalia Elephant silver coin, emphasis on skin texture engraving and metallic edge luster.',
  'A surreal editorial scene of castles built inside a browser sandbox, translucent walls, strict geometric boundaries, glowing checker lines, dramatic side lighting.',
  'A futuristic operations room visualizing proofware: deterministic acceptance and rejection traces projected as layered HUD panels over a dark grid.',
  'A cinematic concept art frame of a local-first AI workstation with WebGPU kernels flowing into verification checkpoints, neon teal accents, restrained composition.',
  'An abstract infographic-style artwork showing interface to reasoning to checker flow as three distinct luminous channels converging into a green accept gate.',
  'A moody Brooklyn night street with wet pavement reflections, matte storefronts, minimal signage, and subtle cyber-industrial atmosphere.',
  'A technical poster aesthetic featuring Lean theorem symbols and circuit motifs, monochrome base with sharp teal highlights, clean negative space.',
]);
const DIFFUSION_NEGATIVE_STARTER_PROMPTS = Object.freeze([
  'blurry, lowres, jpeg artifacts, noisy, text, watermark',
  'deformed anatomy, extra fingers, duplicated limbs, bad hands',
  'overexposed, underexposed, washed colors, poor contrast',
  'cropped subject, out of frame, tilted horizon',
  'cartoonish proportions, unrealistic shadows, flat lighting',
  'muddy details, over-smoothing, plastic skin',
  'logo, signature, timestamp, subtitles',
  'distorted perspective, warped geometry, stretched objects',
  'banding, posterization, chromatic aberration',
  'cluttered background, messy composition, visual noise',
  'unreadable typography, gibberish text, malformed letters',
  'double pupils, asymmetrical eyes, broken facial structure',
  'incorrect limb count, fused fingers, disconnected joints',
  'overprocessed HDR, halo edges, ringing artifacts',
  'flat depth, no focal separation, poor subject isolation',
  'compression blocks, aliasing, moire patterns, scan lines',
]);

function normalizeQuickModeToken(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'run' || normalized === 'text') return 'run';
  if (normalized === 'embedding' || normalized === 'embed') return 'embedding';
  if (normalized === 'diffusion' || normalized === 'image') return 'diffusion';
  if (normalized === 'energy') return 'energy';
  return null;
}

function normalizeQuickModes(rawMode, rawModes) {
  const values = [];
  if (Array.isArray(rawModes)) values.push(...rawModes);
  if (rawMode !== undefined) values.push(rawMode);
  const tokens = new Set();
  for (const value of values) {
    if (typeof value === 'string') {
      const lowered = value.trim().toLowerCase();
      if (lowered === 'both' || lowered === 'all' || lowered === 'text+embedding') {
        tokens.add('run');
        tokens.add('embedding');
        continue;
      }
      const splitValues = lowered.split(/[,\s+/]+/).filter(Boolean);
      for (const token of splitValues) {
        const normalized = normalizeQuickModeToken(token);
        if (normalized) tokens.add(normalized);
      }
      continue;
    }
    const normalized = normalizeQuickModeToken(value);
    if (normalized) tokens.add(normalized);
  }
  if (tokens.size === 0) {
    tokens.add('run');
  }
  return [...tokens];
}

function resolveQuickModelBaseUrl(baseUrl, modelId) {
  if (typeof baseUrl === 'string' && baseUrl.trim()) {
    return new URL(baseUrl.trim(), QUICK_MODEL_CATALOG_URL).toString();
  }
  const encoded = encodeURIComponent(modelId);
  return new URL(`./curated/${encoded}`, QUICK_MODEL_CATALOG_URL).toString();
}

function normalizeQuickCatalogEntry(raw, index) {
  if (!raw || typeof raw !== 'object') return null;
  const modelId = typeof raw.modelId === 'string' ? raw.modelId.trim() : '';
  if (!modelId) return null;
  const modes = normalizeQuickModes(raw.mode, raw.modes);
  const sizeBytes = Number(raw.sizeBytes);
  return {
    id: modelId,
    modelId,
    label: typeof raw.label === 'string' && raw.label.trim() ? raw.label.trim() : modelId,
    description: typeof raw.description === 'string' ? raw.description.trim() : '',
    baseUrl: resolveQuickModelBaseUrl(raw.baseUrl, modelId),
    modes,
    sizeBytes: Number.isFinite(sizeBytes) && sizeBytes > 0 ? Math.floor(sizeBytes) : null,
    recommended: raw.recommended === true,
    sortOrder: Number.isFinite(Number(raw.sortOrder)) ? Number(raw.sortOrder) : index,
  };
}

function parseQuickCatalogPayload(payload) {
  if (!payload || typeof payload !== 'object') {
    return [];
  }
  const entries = Array.isArray(payload.models) ? payload.models : [];
  const normalized = [];
  for (let i = 0; i < entries.length; i += 1) {
    const entry = normalizeQuickCatalogEntry(entries[i], i);
    if (!entry) continue;
    normalized.push(entry);
  }
  normalized.sort((a, b) => {
    if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
    if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
    return a.label.localeCompare(b.label);
  });
  return normalized;
}

function getQuickCatalogEntries() {
  return Array.isArray(state.quickModelCatalog) ? state.quickModelCatalog : [];
}

function formatQuickModelBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'size unknown';
  return formatBytes(bytes);
}

function formatDownloadMegabytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return '0.0';
  return (value / (1024 * 1024)).toFixed(1);
}

function resolveDownloadProgressForModel(modelId) {
  const progress = state.downloadProgress;
  if (!progress || typeof progress !== 'object') return null;
  const progressModelId = typeof progress.modelId === 'string' ? progress.modelId : '';
  if (modelId && progressModelId && progressModelId !== modelId) return null;

  const percent = Number(progress.percent);
  const downloadedBytes = Number(progress.downloadedBytes);
  const totalBytes = Number(progress.totalBytes);
  return {
    modelId: progressModelId || modelId || '',
    percent: Number.isFinite(percent) ? clampPercent(percent) : null,
    downloadedBytes: Number.isFinite(downloadedBytes) && downloadedBytes > 0 ? downloadedBytes : 0,
    totalBytes: Number.isFinite(totalBytes) && totalBytes > 0 ? totalBytes : 0,
  };
}

function formatQuickImportProgress(modelId) {
  const progress = resolveDownloadProgressForModel(modelId);
  if (!progress) return '';
  const parts = [];
  if (Number.isFinite(progress.percent)) {
    parts.push(`${progress.percent.toFixed(1)}%`);
  }
  if (progress.totalBytes > 0) {
    const done = formatDownloadMegabytes(progress.downloadedBytes);
    const total = formatDownloadMegabytes(progress.totalBytes);
    parts.push(`${done}/${total} MB`);
  } else if (progress.downloadedBytes > 0) {
    parts.push(`${formatDownloadMegabytes(progress.downloadedBytes)} MB`);
  }
  return parts.join(' · ');
}

function findQuickModelEntry(modelId) {
  return getQuickCatalogEntries().find((entry) => entry.modelId === modelId) || null;
}

function formatQuickModelModeBadge(modes = []) {
  if (!Array.isArray(modes) || modes.length === 0) return 'text';
  const labels = [];
  if (modes.includes('run')) labels.push('text');
  if (modes.includes('embedding')) labels.push('embedding');
  if (modes.includes('diffusion')) labels.push('diffusion');
  if (modes.includes('energy')) labels.push('energy');
  return labels.length > 0 ? labels.join('+') : 'text';
}

function getComparableQuickModelSize(entry) {
  const size = Number(entry?.sizeBytes);
  return Number.isFinite(size) && size > 0 ? size : Number.POSITIVE_INFINITY;
}

function getSmallestQuickModelForMode(modeToken) {
  if (!modeToken) return null;
  const candidates = getQuickCatalogEntries().filter((entry) => entry.modes.includes(modeToken));
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => {
    const sizeDiff = getComparableQuickModelSize(a) - getComparableQuickModelSize(b);
    if (sizeDiff !== 0) return sizeDiff;
    if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
    return a.label.localeCompare(b.label);
  });
  return candidates[0] || null;
}

function getDiagnosticsRequiredQuickMode() {
  const selection = state.diagnosticsSelections?.diagnostics || {};
  const selectedProfile = decodeDiagnosticsProfileId(selection.profile || '');
  const suite = selectedProfile?.suite || selection.suite || getDiagnosticsDefaultSuite('diagnostics');
  if (suite === 'kernels') return null;
  if (suite === 'diffusion') return 'diffusion';
  if (suite === 'energy') return 'energy';
  const preset = String(selectedProfile?.preset || selection.preset || '').toLowerCase();
  if (preset.includes('embedding')) return 'embedding';
  return 'run';
}

function updateNavState(mode) {
  // Treat the top 5 buttons as a single selection control:
  // exactly one of {run,diffusion,energy,diagnostics,models} is active.
  const normalizedMode = mode === 'kernels' ? 'diagnostics' : mode;
  const isPrimary = PRIMARY_MODES.has(normalizedMode);

  document.querySelectorAll('.mode-tab').forEach((button) => {
    const isActive = isPrimary && button.dataset.mode === normalizedMode;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
  document.querySelectorAll('.mode-tool').forEach((button) => {
    const isActive = !isPrimary && button.dataset.mode === normalizedMode;
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

function ensurePrimaryModeControlStack() {
  const panelGrid = $('panel-grid');
  if (!panelGrid) return;

  const railStack = panelGrid.querySelector('.panel-stack-rail');
  if (!railStack) return;

  let controlsStack = panelGrid.querySelector('.panel-stack-controls');
  if (!controlsStack) {
    controlsStack = document.createElement('div');
    controlsStack.className = 'panel-stack panel-stack-controls';
    controlsStack.dataset.modes = 'run embedding diffusion energy';
    panelGrid.insertBefore(controlsStack, railStack);
  }

  const controlSectionSelectors = [
    '.run-controls-panel',
    '.diffusion-controls-panel',
    '.energy-controls-panel',
    '.energy-solver-panel',
  ];
  for (const selector of controlSectionSelectors) {
    const section = panelGrid.querySelector(selector);
    if (!section || section.parentElement === controlsStack) continue;
    controlsStack.appendChild(section);
  }
}

function syncRunModeUI(mode) {
  const isEmbeddingMode = mode === 'embedding';
  setText($('run-panel-title'), isEmbeddingMode ? 'Embeddings' : 'Text Decoding');
  setText($('run-controls-title'), isEmbeddingMode ? 'Embedding Controls' : 'Run Controls');
  setText($('run-prompt-label'), isEmbeddingMode ? 'Input text' : 'Prompt');
  setText($('run-generate-btn'), isEmbeddingMode ? 'Embed' : 'Generate');
  const prompt = $('run-prompt');
  if (prompt) {
    prompt.placeholder = isEmbeddingMode
      ? 'Enter text to embed...'
      : 'Ask a question or provide a prompt...';
  }
  setHidden($('run-sampling-controls'), isEmbeddingMode);
  setHidden($('run-embedding-docs'), !isEmbeddingMode);
  if (isEmbeddingMode) {
    refreshEmbeddingDemoDocuments();
  }
  renderEmbeddingDocumentSet();
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
  syncRunModeUI(mode);
  syncDiagnosticsModeUI(mode);
  updateModelEmptyStates();
  updatePerformancePanel();
  renderRunLog();
  if (mode === 'models') {
    refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onTryModel: handleStorageTryModel,
      onUnloadActiveModel: unloadActivePipeline,
      onStorageInventoryRefreshed: renderQuickModelPanels,
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

function getModelAvailability() {
  const availability = state.modelAvailability;
  if (!availability || typeof availability !== 'object') {
    return { ...DEFAULT_MODEL_AVAILABILITY };
  }
  return {
    total: Number.isFinite(availability.total) ? availability.total : 0,
    run: Number.isFinite(availability.run) ? availability.run : 0,
    embedding: Number.isFinite(availability.embedding) ? availability.embedding : 0,
    diffusion: Number.isFinite(availability.diffusion) ? availability.diffusion : 0,
    energy: Number.isFinite(availability.energy) ? availability.energy : 0,
  };
}

function setEmptyNotice(scope, message) {
  const notice = $(`${scope}-empty-notice`);
  const text = $(`${scope}-empty-notice-text`);
  const normalized = typeof message === 'string' ? message.trim() : '';
  setHidden(notice, normalized.length === 0);
  setText(text, normalized);
}

function setEmptyNoticeAction(scope, quickModelEntry) {
  const button = $(`${scope}-empty-notice-btn`);
  if (!button) return;
  const busyModelId = state.quickModelActionModelId;
  const hasBusyImport = typeof busyModelId === 'string' && busyModelId.length > 0;

  if (quickModelEntry?.modelId) {
    const isBusy = busyModelId === quickModelEntry.modelId;
    const progressLabel = isBusy ? formatQuickImportProgress(quickModelEntry.modelId) : '';
    button.dataset.noticeAction = 'download';
    button.dataset.quickModelId = quickModelEntry.modelId;
    button.textContent = isBusy
      ? (progressLabel ? `Importing ${progressLabel}` : 'Importing...')
      : `Download ${quickModelEntry.label}`;
    button.disabled = isBusy || (hasBusyImport && !isBusy);
    return;
  }

  button.dataset.noticeAction = 'models';
  delete button.dataset.quickModelId;
  button.textContent = 'Go to Models';
  button.disabled = hasBusyImport;
}

function getMissingModelMessage(mode, availability) {
  const total = Number.isFinite(availability?.total) ? availability.total : 0;
  if (total <= 0) {
    return 'No models found in OPFS. Import a model from the Models tab.';
  }
  const compatible = Number.isFinite(availability?.[mode]) ? availability[mode] : 0;
  if (compatible > 0) return '';
  if (mode === 'embedding') {
    return 'No embedding model available in OPFS for this mode.';
  }
  if (mode === 'diffusion') {
    return 'No diffusion model available in OPFS for this mode.';
  }
  if (mode === 'energy') {
    return 'No energy model available in OPFS for this mode.';
  }
  return 'No text model available in OPFS for this mode.';
}

function setQuickModelStatus(message) {
  const statusEl = $('models-quick-models-status');
  if (!statusEl) return;
  setText(statusEl, message || '');
}

function createQuickModelBadge(text) {
  const badge = document.createElement('span');
  badge.className = 'quick-model-badge';
  badge.textContent = text;
  return badge;
}

function createQuickModelActionButton({ label, action, modelId, disabled, title = '' }) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn btn-small';
  button.textContent = label;
  button.dataset.quickAction = action;
  button.dataset.quickModelId = modelId;
  if (title) {
    button.title = title;
  }
  button.disabled = disabled;
  return button;
}

function renderQuickModelList(listEl, entries) {
  if (!listEl) return;
  listEl.textContent = '';
  const busyId = state.quickModelActionModelId;
  const hasBusyAction = typeof busyId === 'string' && busyId.length > 0;
  const storageIds = new Set(Array.isArray(state.quickModelStorageIds) ? state.quickModelStorageIds : []);

  for (const entry of entries) {
    const isBusy = hasBusyAction && busyId === entry.modelId;
    const isInOpfs = storageIds.has(entry.modelId);

    const card = document.createElement('article');
    card.className = entry.recommended ? 'quick-model-card is-recommended' : 'quick-model-card';

    const row = document.createElement('div');
    row.className = 'quick-model-row';

    const main = document.createElement('div');
    main.className = 'quick-model-main';

    const title = document.createElement('div');
    title.className = 'quick-model-title';
    title.textContent = entry.label;
    main.appendChild(title);

    const modelId = document.createElement('div');
    modelId.className = 'quick-model-id type-caption';
    modelId.textContent = entry.modelId;
    main.appendChild(modelId);

    const meta = document.createElement('div');
    meta.className = 'quick-model-meta';
    if (entry.recommended) {
      meta.appendChild(createQuickModelBadge('recommended'));
    }
    meta.appendChild(createQuickModelBadge(formatQuickModelModeBadge(entry.modes)));
    meta.appendChild(createQuickModelBadge(formatQuickModelBytes(entry.sizeBytes)));
    if (isInOpfs) {
      meta.appendChild(createQuickModelBadge('in opfs'));
    }
    main.appendChild(meta);

    const actions = document.createElement('div');
    actions.className = 'quick-model-actions';
    if (isInOpfs) {
      const imported = document.createElement('span');
      imported.className = 'quick-model-imported type-caption';
      imported.textContent = 'Imported';
      actions.appendChild(imported);
    } else {
      const busyLabel = isBusy ? formatQuickImportProgress(entry.modelId) : '';
      actions.appendChild(createQuickModelActionButton({
        label: isBusy ? (busyLabel ? `Importing ${busyLabel}` : 'Importing...') : 'Import',
        action: 'download',
        modelId: entry.modelId,
        disabled: isBusy || hasBusyAction,
      }));
    }

    row.appendChild(main);
    row.appendChild(actions);
    card.appendChild(row);

    listEl.appendChild(card);
  }

  if (entries.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'type-caption';
    empty.textContent = 'No quick models are configured yet.';
    listEl.appendChild(empty);
  }
}

function renderQuickModelPanels() {
  const catalog = getQuickCatalogEntries();

  if (state.quickModelActionModelId) {
    const modelId = state.quickModelActionModelId;
    const progressLabel = formatQuickImportProgress(modelId);
    setQuickModelStatus(progressLabel ? `Importing ${modelId}: ${progressLabel}` : `Importing ${modelId}...`);
  } else if (state.quickModelCatalogLoading) {
    setQuickModelStatus('Loading quick models...');
  } else if (state.quickModelCatalogError) {
    const message = `Quick model catalog unavailable: ${state.quickModelCatalogError}`;
    setQuickModelStatus(message);
  } else {
    setQuickModelStatus(
      catalog.length > 0
        ? ''
        : 'No quick models configured in catalog.json yet.'
    );
  }

  renderQuickModelList($('models-quick-models-list'), catalog);
}

async function loadQuickModelCatalog() {
  state.quickModelCatalogLoading = true;
  state.quickModelCatalogError = null;
  renderQuickModelPanels();
  try {
    const response = await fetch(QUICK_MODEL_CATALOG_URL, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    state.quickModelCatalog = parseQuickCatalogPayload(payload);
  } catch (error) {
    state.quickModelCatalog = [];
    state.quickModelCatalogError = error instanceof Error ? error.message : String(error);
  } finally {
    state.quickModelCatalogLoading = false;
    renderQuickModelPanels();
  }
}

async function applyImportedModelToCurrentMode(modelId) {
  if (!modelId) return;
  const mode = state.uiMode;
  if (mode === 'models') return;

  if (mode === 'diagnostics') {
    selectDiagnosticsModel(modelId);
    return;
  }

  if (!isModeModelSelectable(mode)) return;
  const modelType = await getModelTypeForId(modelId);
  if (!isCompatibleModelType(modelType, mode)) return;

  selectDiagnosticsModel(modelId);
  state.modeModelId[mode] = modelId;
}

async function handleEmptyNoticeAction(scope) {
  const button = $(`${scope}-empty-notice-btn`);
  if (!button) return;
  const action = button.dataset.noticeAction || 'models';
  if (action !== 'download') {
    setUiMode('models');
    return;
  }
  const modelId = button.dataset.quickModelId || '';
  if (!modelId) {
    setUiMode('models');
    return;
  }
  await runQuickModelAction('download', modelId);
}

function handleDownloadProgressEvent(progress) {
  const modelId = typeof progress?.modelId === 'string' && progress.modelId.trim()
    ? progress.modelId.trim()
    : (typeof state.activeDownloadId === 'string' ? state.activeDownloadId : '');
  const percent = Number(progress?.percent);
  const downloadedBytes = Number(progress?.downloadedBytes);
  const totalBytes = Number(progress?.totalBytes);

  state.downloadProgress = {
    modelId,
    percent: Number.isFinite(percent) ? clampPercent(percent) : null,
    downloadedBytes: Number.isFinite(downloadedBytes) && downloadedBytes > 0 ? downloadedBytes : 0,
    totalBytes: Number.isFinite(totalBytes) && totalBytes > 0 ? totalBytes : 0,
    status: typeof progress?.status === 'string' ? progress.status : '',
  };
  if (modelId) {
    state.activeDownloadId = modelId;
  }
  state.downloadActive = true;
  updateStatusIndicator();
  if (state.quickModelActionModelId && modelId && modelId === state.quickModelActionModelId) {
    updateModelEmptyStates();
  } else {
    renderQuickModelPanels();
  }
}

function handleDownloadStateChangeEvent(update) {
  if (!update || typeof update !== 'object') return;
  const modelId = typeof update.modelId === 'string' && update.modelId.trim() ? update.modelId.trim() : '';
  if (modelId) {
    state.activeDownloadId = modelId;
  }
  if (update.active === true) {
    state.downloadActive = true;
  } else if (update.active === false) {
    state.downloadActive = false;
    if (!modelId || state.downloadProgress?.modelId === modelId) {
      state.downloadProgress = null;
    }
  }
  updateStatusIndicator();
  if (state.quickModelActionModelId && (!modelId || modelId === state.quickModelActionModelId)) {
    updateModelEmptyStates();
  } else {
    renderQuickModelPanels();
  }
}

async function runQuickModelAction(action, modelId) {
  if (action !== 'download') return;
  const entry = findQuickModelEntry(modelId);
  if (!entry) {
    updateConvertStatus(`Quick model not found: ${modelId}`, 0);
    return;
  }
  if (state.quickModelActionModelId) return;

  let finalQuickStatus = '';
  state.quickModelActionModelId = modelId;
  state.downloadActive = true;
  state.activeDownloadId = modelId;
  state.downloadProgress = null;
  updateStatusIndicator();
  setQuickModelStatus(`Importing ${modelId}...`);
  updateModelEmptyStates();
  renderQuickModelPanels();
  try {
    const imported = await startDownloadFromBaseUrl(entry.baseUrl, entry.modelId);
    if (!imported) {
      throw new Error(`Could not import model ${modelId}.`);
    }
    await updateStorageInfo();
    await refreshModelList();
    await applyImportedModelToCurrentMode(modelId);
    if (state.uiMode === 'models') {
      await refreshStorageInspector({
        onSelectModel: selectDiagnosticsModel,
        onTryModel: handleStorageTryModel,
        onUnloadActiveModel: unloadActivePipeline,
        onStorageInventoryRefreshed: renderQuickModelPanels,
        onModelsUpdated: refreshModelList,
      });
    }
    finalQuickStatus = `Imported ${modelId} to OPFS.`;
    renderQuickModelPanels();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    finalQuickStatus = `Import failed: ${message}`;
    updateConvertStatus(`Quick model action failed: ${message}`, 0);
    updateDiagnosticsStatus(`Quick model action failed: ${message}`, true);
  } finally {
    if (!state.downloadProgress || state.downloadProgress.modelId === modelId) {
      state.downloadProgress = null;
    }
    state.quickModelActionModelId = null;
    state.downloadActive = false;
    state.activeDownloadId = null;
    updateStatusIndicator();
    updateModelEmptyStates();
    renderQuickModelPanels();
    if (finalQuickStatus) {
      setQuickModelStatus(finalQuickStatus);
    }
  }
}

function updateModelEmptyStates() {
  const availability = getModelAvailability();
  const runTargetMode = state.uiMode === 'embedding' ? 'embedding' : 'run';
  const runMessage = getMissingModelMessage(runTargetMode, availability);
  const diffusionMessage = getMissingModelMessage('diffusion', availability);
  const energyMessage = getMissingModelMessage('energy', availability);
  const diagnosticsTargetMode = getDiagnosticsRequiredQuickMode();
  const diagnosticsMessage = (
    state.uiMode === 'diagnostics'
      ? (diagnosticsTargetMode ? getMissingModelMessage(diagnosticsTargetMode, availability) : '')
      : ''
  );

  setEmptyNotice('run', runMessage);
  setEmptyNotice('diffusion', diffusionMessage);
  setEmptyNotice('energy', energyMessage);
  setEmptyNotice('diagnostics', diagnosticsMessage);
  setEmptyNoticeAction('run', runMessage ? getSmallestQuickModelForMode(runTargetMode) : null);
  setEmptyNoticeAction('diffusion', diffusionMessage ? getSmallestQuickModelForMode('diffusion') : null);
  setEmptyNoticeAction('energy', energyMessage ? getSmallestQuickModelForMode('energy') : null);
  setEmptyNoticeAction('diagnostics', diagnosticsMessage ? getSmallestQuickModelForMode(diagnosticsTargetMode) : null);
  renderQuickModelPanels();

  const diffusionRun = $('diffusion-run-btn');
  if (diffusionRun) {
    diffusionRun.disabled = state.diffusionGenerating || state.diffusionLoading || diffusionMessage.length > 0;
  }
  const energyRun = $('energy-run-btn');
  if (energyRun) {
    energyRun.disabled = state.energyGenerating || state.energyLoading || energyMessage.length > 0;
  }
  syncRunControls();
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

const AUX_IMPORT_FILENAMES = [
  'config.json',
  'generation_config.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'added_tokens.json',
  'preprocessor_config.json',
  'vocab.txt',
  'merges.txt',
];

function getPickedFilePath(file) {
  if (!file) return '';
  if (typeof file.relativePath === 'string' && file.relativePath.length > 0) {
    return file.relativePath;
  }
  if (typeof file.webkitRelativePath === 'string' && file.webkitRelativePath.length > 0) {
    return file.webkitRelativePath;
  }
  if (typeof file.name === 'string') return file.name;
  return '';
}

function normalizePickedPath(path) {
  return String(path || '')
    .replace(/\\/g, '/')
    .replace(/^\.?\//, '')
    .trim();
}

function getPathBaseName(path) {
  const normalized = normalizePickedPath(path);
  if (!normalized) return '';
  const parts = normalized.split('/');
  return parts[parts.length - 1] || '';
}

function findPickedFileByPath(files, path) {
  const targetPath = normalizePickedPath(path);
  if (!targetPath) return null;

  const exact = files.find((file) => normalizePickedPath(getPickedFilePath(file)) === targetPath);
  if (exact) return exact;

  const targetBase = getPathBaseName(targetPath);
  if (!targetBase) return null;
  const baseMatches = files.filter((file) => getPathBaseName(getPickedFilePath(file)) === targetBase);
  if (baseMatches.length === 1) return baseMatches[0];
  return null;
}

function findPickedFileByBaseName(files, name) {
  const target = String(name || '').trim();
  if (!target) return null;
  const matches = files.filter((file) => getPathBaseName(getPickedFilePath(file)) === target);
  if (matches.length === 0) return null;
  return matches[0];
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
  if (!compatibleId) {
    state.modeModelId[mode] = null;
    if (state.uiMode === mode) {
      state.activeModelId = null;
      const modelSelect = $('diagnostics-model');
      if (modelSelect) modelSelect.value = '';
    }
    return;
  }
  if (state.activeModelId !== compatibleId) {
    if (state.activePipeline && state.activePipelineModelId && state.activePipelineModelId !== compatibleId) {
      await unloadActivePipeline();
    }
    selectDiagnosticsModel(compatibleId);
  }
  state.modeModelId[mode] = compatibleId;
}

function getUiModeForModelType(modelType) {
  const normalizedType = normalizeModelType(modelType);
  if (normalizedType === 'embedding') return 'embedding';
  if (normalizedType === 'diffusion') return 'diffusion';
  if (normalizedType === 'energy') return 'energy';
  return 'run';
}

async function handleStorageTryModel(modelId) {
  if (!modelId) return;
  const modelType = await getModelTypeForId(modelId);
  const targetMode = getUiModeForModelType(modelType);
  setUiMode(targetMode);
  await refreshModelList();
  selectDiagnosticsModel(modelId);
}

function updateSidebarLayout(models) {
  const panelGrid = $('panel-grid');
  if (!panelGrid) return;
  const hasModels = Array.isArray(models) && models.length > 0;
  panelGrid.dataset.layout = hasModels ? 'ready' : 'empty';
}

async function computeModelAvailability(models) {
  const availability = { ...DEFAULT_MODEL_AVAILABILITY };
  if (!Array.isArray(models)) return availability;
  const seenModelIds = new Set();
  for (const model of models) {
    const modelId = typeof model?.modelId === 'string' && model.modelId
      ? model.modelId
      : (typeof model?.id === 'string' ? model.id : '');
    if (!modelId || seenModelIds.has(modelId)) continue;
    seenModelIds.add(modelId);
    availability.total += 1;

    let modelType = normalizeModelType(model?.modelType);
    if (!modelType) {
      modelType = normalizeModelType(await getModelTypeForId(modelId));
    }
    if (isCompatibleModelType(modelType, 'run')) availability.run += 1;
    if (isCompatibleModelType(modelType, 'embedding')) availability.embedding += 1;
    if (isCompatibleModelType(modelType, 'diffusion')) availability.diffusion += 1;
    if (isCompatibleModelType(modelType, 'energy')) availability.energy += 1;
  }
  return availability;
}

async function refreshModelList() {
  const modelSelect = $('diagnostics-model');
  if (!modelSelect) return;
  const refreshVersion = ++modelListRefreshVersion;
  let models = [];
  try {
    models = await listRegisteredModels();
  } catch (error) {
    log.warn('DopplerDemo', `Model registry unavailable: ${error.message}`);
  }
  state.registeredModelIds = [...new Set(models
    .map((entry) => {
      if (typeof entry?.modelId === 'string' && entry.modelId) return entry.modelId;
      if (typeof entry?.id === 'string' && entry.id) return entry.id;
      return '';
    })
    .filter(Boolean))];
  const filteredModels = await filterModelsForMode(models, state.uiMode);
  if (refreshVersion !== modelListRefreshVersion) return;
  modelSelect.innerHTML = '';
  const modelIds = [];
  const seenModelIds = new Set();
  for (const model of filteredModels) {
    const modelId = typeof model?.modelId === 'string' && model.modelId
      ? model.modelId
      : (typeof model?.id === 'string' ? model.id : '');
    if (!modelId || seenModelIds.has(modelId)) continue;
    const entryModelType = normalizeModelType(model?.modelType);
    if (entryModelType) {
      state.modelTypeCache[modelId] = entryModelType;
    }
    seenModelIds.add(modelId);
    modelIds.push(modelId);
  }
  if (!modelIds.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = `No ${getModeModelLabel(state.uiMode)} models`;
    modelSelect.appendChild(opt);
  } else {
    for (const modelId of modelIds) {
      const opt = document.createElement('option');
      opt.value = modelId;
      opt.textContent = modelId;
      modelSelect.appendChild(opt);
    }
  }
  updateSidebarLayout(models);
  state.modelAvailability = await computeModelAvailability(models);
  await updateStorageInfo();
  await syncModelForMode(state.uiMode);
  updateModelEmptyStates();
  updateDiagnosticsGuidance();
  if (state.uiMode === 'energy') {
    await preloadEnergyPipelineIfNeeded();
  }
  if (state.uiMode === 'models') {
    await refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onTryModel: handleStorageTryModel,
      onUnloadActiveModel: unloadActivePipeline,
      onStorageInventoryRefreshed: renderQuickModelPanels,
      onModelsUpdated: refreshModelList,
    });
    renderQuickModelPanels();
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

function pickRandomStarter(pool) {
  if (!Array.isArray(pool) || pool.length === 0) return '';
  const index = Math.floor(Math.random() * pool.length);
  return String(pool[index] || '').trim();
}

function isStarterExampleInput(inputEl) {
  return inputEl?.dataset?.starterExample === '1';
}

function setStarterExampleInput(inputEl, isExample) {
  if (!inputEl) return;
  inputEl.dataset.starterExample = isExample ? '1' : '0';
}

function pickRandomStarterDifferent(pool, currentValue) {
  if (!Array.isArray(pool) || pool.length === 0) return '';
  const current = String(currentValue || '').trim();
  if (pool.length === 1) return String(pool[0] || '').trim();
  for (let attempt = 0; attempt < pool.length * 2; attempt += 1) {
    const next = pickRandomStarter(pool);
    if (next && next !== current) {
      return next;
    }
  }
  return pickRandomStarter(pool);
}

function pickRandomSubset(pool, count) {
  if (!Array.isArray(pool) || pool.length === 0) return [];
  const targetCount = Math.max(1, Math.min(Number(count) || 1, pool.length));
  const copy = pool.slice();
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, targetCount);
}

function refreshEmbeddingDemoDocuments(options = {}) {
  const { force = false } = options;
  const current = Array.isArray(state.embeddingDemoDocuments) ? state.embeddingDemoDocuments : [];
  if (!force && current.length === EMBEDDING_DEMO_DOCUMENT_COUNT) {
    return current;
  }
  state.embeddingDemoDocuments = pickRandomSubset(
    EMBEDDING_DEMO_DOCUMENT_CATALOG,
    EMBEDDING_DEMO_DOCUMENT_COUNT
  );
  renderEmbeddingDocumentSet();
  return state.embeddingDemoDocuments;
}

function renderEmbeddingDocumentSet() {
  const wrap = $('run-embedding-docs');
  const list = $('run-embedding-docs-list');
  if (!wrap || !list) return;
  if (state.uiMode !== 'embedding') {
    setHidden(wrap, true);
    return;
  }
  setHidden(wrap, false);
  const docs = Array.isArray(state.embeddingDemoDocuments) ? state.embeddingDemoDocuments : [];
  if (docs.length === 0) {
    list.innerHTML = '<div class="type-caption">No documents configured.</div>';
    return;
  }
  const rows = docs
    .map((doc, index) => {
      const text = String(doc?.text || '').trim();
      const snippet = text.length > 140 ? `${text.slice(0, 140)}...` : text;
      return `<div class="embedding-doc-item"><div class="type-caption"><strong>${index + 1}. ${doc.title}</strong></div><div class="type-caption">${snippet}</div></div>`;
    })
    .join('');
  list.innerHTML = rows;
}

function applyStarterPrompt(inputEl, pool, options = {}) {
  if (!inputEl) return;
  const { force = false } = options;
  const current = String(inputEl.value || '').trim();
  if (!force && current.length > 0) return;
  const next = force ? pickRandomStarterDifferent(pool, current) : pickRandomStarter(pool);
  if (!next) return;
  inputEl.value = next;
  setStarterExampleInput(inputEl, true);
}

function prefillDemoTextInputs() {
  applyStarterPrompt($('run-prompt'), RUN_STARTER_PROMPTS);
  applyStarterPrompt($('diffusion-prompt'), DIFFUSION_STARTER_PROMPTS);
  applyStarterPrompt($('diffusion-negative'), DIFFUSION_NEGATIVE_STARTER_PROMPTS);
}

function bindStarterPromptInput(inputEl) {
  if (!inputEl) return;
  inputEl.addEventListener('focus', () => {
    if (isStarterExampleInput(inputEl)) {
      inputEl.select();
    }
  });
  inputEl.addEventListener('input', () => {
    setStarterExampleInput(inputEl, false);
  });
}

function syncRunControls() {
  const runPrompt = $('run-prompt');
  const runGenerate = $('run-generate-btn');
  const runStop = $('run-stop-btn');
  const runClear = $('run-clear-btn');
  const runResetKvToggle = $('run-reset-kv-toggle');
  const temperatureInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const maxTokensInput = $('max-tokens-input');
  const availability = getModelAvailability();
  const needsEmbeddingModel = state.uiMode === 'embedding';
  const hasCompatibleModel = needsEmbeddingModel ? availability.embedding > 0 : availability.run > 0;
  const disabled = state.runGenerating || state.runLoading;
  if (runPrompt) runPrompt.disabled = disabled;
  if (runGenerate) runGenerate.disabled = disabled || !hasCompatibleModel;
  if (runClear) runClear.disabled = disabled;
  if (runResetKvToggle) runResetKvToggle.disabled = disabled;
  if (temperatureInput) temperatureInput.disabled = disabled;
  if (topPInput) topPInput.disabled = disabled;
  if (topKInput) topKInput.disabled = disabled;
  if (maxTokensInput) maxTokensInput.disabled = disabled;
  if (runStop) setHidden(runStop, !state.runGenerating);
}

function setRunGenerating(isGenerating) {
  state.runGenerating = Boolean(isGenerating);
  if (!state.runGenerating) {
    state.runPrefilling = false;
  }
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
  if (state.uiMode === 'embedding') {
    return {};
  }
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
    const runMode = state.uiMode === 'embedding' ? 'embedding' : 'run';
    state.modeModelId[runMode] = modelId;
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

async function preloadEnergyPipelineIfNeeded() {
  if (state.uiMode !== 'energy') return;
  if (state.energyLoading || state.energyGenerating) return;

  const modelId = getSelectedModelId();
  if (!modelId) return;

  const selectedModelType = normalizeModelType(await getModelTypeForId(modelId));
  if (selectedModelType !== 'energy') return;

  const activeModelType = normalizeModelType(state.activePipeline?.manifest?.modelType);
  if (
    state.activePipeline &&
    state.activeModelId === modelId &&
    activeModelType === 'energy'
  ) {
    return;
  }

  updateEnergyStatus('Loading energy model...');
  try {
    await ensureEnergyPipeline();
    if (!state.energyGenerating) updateEnergyStatus('Ready');
  } catch (error) {
    log.warn('DopplerDemo', `Energy preload skipped: ${error.message}`);
    if (!state.energyGenerating) updateEnergyStatus('Idle');
  }
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
  const isEmbeddingMode = state.uiMode === 'embedding';
  const runResetKvToggle = $('run-reset-kv-toggle');
  const resetContextEachRun = !isEmbeddingMode && Boolean(runResetKvToggle?.checked);
  if (!prompt) {
    updateRunStatus(isEmbeddingMode ? 'Enter text to embed.' : 'Enter a prompt to generate.');
    return;
  }

  updateRunStatus('Preparing...');
  let pipeline;
  let modelType = null;
  try {
    pipeline = await ensureRunPipeline();
    modelType = normalizeModelType(pipeline?.manifest?.modelType);
    if (isEmbeddingMode && modelType !== 'embedding') {
      throw new Error('Selected model is not an embedding model.');
    }
    if (!isEmbeddingMode && (modelType === 'diffusion' || modelType === 'energy' || modelType === 'embedding')) {
      throw new Error('Selected model is not a text model.');
    }
    if (resetContextEachRun) {
      pipeline.reset?.();
    }
  } catch (error) {
    updateRunStatus(`Error: ${error.message}`);
    return;
  }

  const controller = new AbortController();
  state.runAbortController = controller;
  state.runPrefilling = !isEmbeddingMode;
  setRunGenerating(true);
  updateRunStatus(isEmbeddingMode ? 'Embedding...' : 'Generating...');
  if (outputEl) outputEl.textContent = '';

  const options = buildRunGenerateOptions();
  const isEmbeddingModel = modelType === 'embedding';
  let output = '';
  let tokenCount = 0;
  const start = performance.now();
  let firstTokenAt = null;

  try {
    if (isEmbeddingModel) {
      const embedStart = performance.now();
      pipeline.reset?.();
      const result = await pipeline.embed(prompt, options);
      const queryEmbeddingValues = result?.embedding ?? new Float32Array(0);
      const querySummary = summarizeEmbeddingVector(queryEmbeddingValues);
      if (!Number.isFinite(querySummary.dimension) || querySummary.dimension <= 0) {
        throw new Error('No embedding returned.');
      }
      if (querySummary.nonFiniteCount > 0) {
        throw new Error(`Embedding contains non-finite values (${querySummary.nonFiniteCount}/${querySummary.dimension}).`);
      }
      const embeddingDocuments = refreshEmbeddingDemoDocuments({ force: true });
      updateRunStatus('Embedding demo documents...');
      const scoredDocuments = [];
      for (const doc of embeddingDocuments) {
        pipeline.reset?.();
        const docResult = await pipeline.embed(doc.text, options);
        const docEmbeddingValues = docResult?.embedding ?? new Float32Array(0);
        const docSummary = summarizeEmbeddingVector(docEmbeddingValues);
        const score = cosineSimilarity(queryEmbeddingValues, docEmbeddingValues);
        scoredDocuments.push({
          id: doc.id,
          title: doc.title,
          text: doc.text,
          tokens: Number.isFinite(docResult?.tokens?.length) ? docResult.tokens.length : 0,
          dimension: docSummary.dimension,
          nonFinite: docSummary.nonFiniteCount,
          score: Number.isFinite(score) ? Number(score.toFixed(6)) : null,
        });
      }

      const ranked = scoredDocuments
        .slice()
        .sort((a, b) => (b.score ?? Number.NEGATIVE_INFINITY) - (a.score ?? Number.NEGATIVE_INFINITY))
        .map((entry, index) => ({ rank: index + 1, ...entry }));
      const embeddingMs = Math.max(1, performance.now() - embedStart);

      output = JSON.stringify(
        {
          mode: 'embedding',
          query: prompt,
          dimension: querySummary.dimension,
          tokens: result?.tokens?.length ?? 0,
          embedding_preview: querySummary.preview,
          retrieval: {
            documents: scoredDocuments,
            ranked,
            top_match: ranked[0]
              ? { id: ranked[0].id, title: ranked[0].title, score: ranked[0].score }
              : null,
          },
        },
        null,
        2
      );
      state.lastMetrics = {
        ...(state.lastMetrics || {}),
        embeddingDim: querySummary.dimension,
        embeddingMs: Number(embeddingMs.toFixed(2)),
      };
      if (outputEl) outputEl.textContent = output;
      updateRunStatus('Complete');
    } else {
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
          if (state.runPrefilling) {
            state.runPrefilling = false;
            updateStatusIndicator();
          }
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
    }
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
  if (promptEl) {
    promptEl.value = '';
    setStarterExampleInput(promptEl, false);
  }
  if (outputEl) outputEl.textContent = '';
  updateRunStatus('Idle');
}

function handleInferencePulseReset() {
  state.lastMetrics = null;
  state.lastInferenceStats = null;
  state.lastMemoryStats = null;
  state.lastDiffusionRequest = null;
  state.lastEnergyRequest = null;
  state.runLog = [];
  state.runCounter = 0;

  const snapshot = captureMemorySnapshot();
  updatePerformancePanel(snapshot);
  updateMemoryPanel(snapshot);
  renderRunLog();
}

function summarizeEmbeddingVector(values) {
  const dimension = Number.isFinite(values?.length) ? values.length : 0;
  let nonFiniteCount = 0;
  for (let i = 0; i < dimension; i++) {
    if (!Number.isFinite(values[i])) nonFiniteCount++;
  }
  return {
    dimension,
    nonFiniteCount,
    preview: Array.from(values.slice(0, Math.min(16, dimension))).map((v) => Number(v.toFixed(6))),
  };
}

function cosineSimilarity(a, b) {
  if (!ArrayBuffer.isView(a) || !ArrayBuffer.isView(b)) return null;
  if (a.length !== b.length || a.length <= 0) return null;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const av = Number(a[i]);
    const bv = Number(b[i]);
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return null;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA <= 0 || normB <= 0) return null;
  return dot / Math.sqrt(normA * normB);
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

function restoreParsedManifest(previousManifest) {
  if (previousManifest) {
    setManifest(previousManifest);
    return;
  }
  clearManifest();
}

async function detectRdrrImport(files) {
  const manifestFile = findPickedFileByBaseName(files, 'manifest.json');
  if (!manifestFile) {
    return { kind: 'none' };
  }

  const manifestText = await manifestFile.text();
  const previousManifest = getManifest();
  let manifest;
  try {
    manifest = parseManifest(manifestText);
  } catch (error) {
    return {
      kind: 'invalid',
      reason: `Found manifest.json but it is not a valid RDRR manifest: ${error.message}`,
    };
  } finally {
    restoreParsedManifest(previousManifest);
  }

  const shardFiles = new Map();
  const missing = [];
  for (const shard of manifest.shards || []) {
    const shardFile = findPickedFileByPath(files, shard.filename);
    if (!shardFile) {
      missing.push(shard.filename || `shard_${shard.index}`);
      continue;
    }
    shardFiles.set(shard.index, shardFile);
  }

  if (missing.length > 0) {
    const preview = missing.slice(0, 3).join(', ');
    const suffix = missing.length > 3 ? ` (+${missing.length - 3} more)` : '';
    return {
      kind: 'invalid',
      reason: `Found RDRR manifest, but shard files are missing: ${preview}${suffix}`,
    };
  }

  let tensorsFile = null;
  if (manifest.tensorsFile) {
    tensorsFile = findPickedFileByPath(files, manifest.tensorsFile);
    if (!tensorsFile) {
      return {
        kind: 'invalid',
        reason: `Found RDRR manifest, but missing tensor map file: ${manifest.tensorsFile}`,
      };
    }
  }

  return {
    kind: 'rdrr',
    manifest,
    manifestText,
    manifestFile,
    shardFiles,
    tensorsFile,
  };
}

async function importRdrrFromFiles(files, detection, label) {
  if (!detection || detection.kind !== 'rdrr') {
    throw new Error('RDRR import requires a valid manifest and shard set.');
  }

  const previousManifest = getManifest();
  state.convertActive = true;
  updateStatusIndicator();
  try {
    const manifest = parseManifest(detection.manifestText);
    const modelId = String(manifest.modelId || '').trim();
    if (!modelId) {
      throw new Error('RDRR manifest is missing modelId.');
    }

    await openModelStore(modelId);

    const shards = Array.isArray(manifest.shards) ? manifest.shards : [];
    const totalSteps = shards.length + (manifest.tensorsFile ? 1 : 0) + 2;
    let completed = 0;
    const step = (message) => {
      completed += 1;
      const percent = totalSteps > 0 ? (completed / totalSteps) * 100 : 100;
      updateConvertStatus(label ? `${message} (${label})` : message, percent);
    };

    await saveManifest(JSON.stringify(manifest, null, 2));
    step(`Saved manifest for ${modelId}`);

    if (manifest.tensorsFile) {
      const tensorsFile = detection.tensorsFile || findPickedFileByPath(files, manifest.tensorsFile);
      if (!tensorsFile) {
        throw new Error(`Missing ${manifest.tensorsFile} for RDRR import.`);
      }
      const tensorsText = await tensorsFile.text();
      await saveTensorsToStore(tensorsText);
      step(`Saved ${manifest.tensorsFile}`);
    }

    const tokenizerFilePath = manifest.tokenizer?.file || null;
    let tokenizerJsonFile = tokenizerFilePath ? findPickedFileByPath(files, tokenizerFilePath) : null;
    let tokenizerModelFile = null;
    if (tokenizerJsonFile && getPathBaseName(getPickedFilePath(tokenizerJsonFile)) === 'tokenizer.model') {
      tokenizerModelFile = tokenizerJsonFile;
      tokenizerJsonFile = null;
    }
    if (!tokenizerJsonFile) {
      tokenizerJsonFile = findPickedFileByBaseName(files, 'tokenizer.json');
    }
    if (!tokenizerModelFile) {
      tokenizerModelFile = findPickedFileByBaseName(files, 'tokenizer.model');
    }

    if (tokenizerJsonFile) {
      await saveTokenizer(await tokenizerJsonFile.text());
    }
    if (tokenizerModelFile) {
      await saveTokenizerModel(await tokenizerModelFile.arrayBuffer());
    }

    for (const filename of AUX_IMPORT_FILENAMES) {
      const auxFile = findPickedFileByBaseName(files, filename);
      if (!auxFile) continue;
      await saveAuxFile(filename, await auxFile.arrayBuffer());
    }

    for (let i = 0; i < shards.length; i++) {
      const shard = shards[i];
      const shardFile = detection.shardFiles.get(shard.index) || findPickedFileByPath(files, shard.filename);
      if (!shardFile) {
        throw new Error(`Missing shard file: ${shard.filename}`);
      }
      const data = new Uint8Array(await shardFile.arrayBuffer());
      if (Number.isFinite(shard.size) && data.byteLength !== shard.size) {
        throw new Error(
          `Shard size mismatch for ${shard.filename}: expected ${shard.size} bytes, got ${data.byteLength}`
        );
      }
      await writeShard(shard.index, data, { verify: true });
      step(`Imported shard ${i + 1}/${shards.length}`);
    }

    await registerDownloadedModel(modelId);
    delete state.modelTypeCache[modelId];
    updateConvertStatus(`RDRR import complete: ${modelId}`, 100);
    await refreshModelList();
  } finally {
    restoreParsedManifest(previousManifest);
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
  if (state.convertActive) return;
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

  const rdrrDetection = await detectRdrrImport(files);
  if (rdrrDetection.kind === 'rdrr') {
    updateConvertStatus(
      `Detected pre-converted RDRR package${pickedLabel ? ` in ${pickedLabel}` : ''}. Importing...`,
      0
    );
    await importRdrrFromFiles(files, rdrrDetection, pickedLabel);
    return;
  }
  if (rdrrDetection.kind === 'invalid' && !hasWeights) {
    updateConvertStatus(rdrrDetection.reason, 0);
    return;
  }
  if (rdrrDetection.kind === 'invalid' && hasWeights) {
    log.warn('DopplerDemo', rdrrDetection.reason);
  }

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
  const profileSelect = $('diagnostics-profile');
  const modelSelect = $('diagnostics-model');
  const presetSelect = $('runtime-preset');
  const selections = state.diagnosticsSelections[state.uiMode] || {};
  const selectedProfileId = profileSelect?.value || selections.profile || '';
  const selectedProfile = decodeDiagnosticsProfileId(selectedProfileId);
  const suite = selectedProfile?.suite || selections.suite || getDiagnosticsDefaultSuite(state.uiMode);
  const modelId = modelSelect?.value || null;
  const runtimePreset = selectedProfile?.preset || selections.preset || presetSelect?.value || DEFAULT_RUNTIME_PRESET;
  if (selectedProfile) {
    storeDiagnosticsSelection(state.uiMode, {
      profile: selectedProfileId,
      suite: selectedProfile.suite,
      preset: selectedProfile.preset,
    });
  }
  if (presetSelect && presetSelect.value !== runtimePreset) {
    presetSelect.value = runtimePreset;
  }
  if (profileSelect && selectedProfileId && profileSelect.value !== selectedProfileId) {
    profileSelect.value = selectedProfileId;
  }
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
      const timestamp = new Date().toISOString();
      const report = {
        suite,
        modelId,
        runtimePreset,
        timestamp,
        results: [{ name: 'verify-config', passed: true }],
        durationMs: 0,
        metrics: { verified: true, mode: 'verify' },
        output: { verified: true, message: 'Configuration verified.' },
      };
      state.lastReport = report;
      state.lastReportInfo = null;
      state.lastMetrics = report.metrics;
      state.lastDiagnosticsSuite = suite;
      updateDiagnosticsStatus('Verified');
      updateDiagnosticsReport(timestamp);
      renderDiagnosticsOutput({ suite, modelId, report }, suite, false);
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
    const message = error instanceof Error ? error.message : String(error);
    updateDiagnosticsStatus(message, true);
    const timestamp = new Date().toISOString();
    const report = {
      suite,
      modelId,
      runtimePreset,
      timestamp,
      results: [{ name: mode === 'verify' ? 'verify-config' : 'run', passed: false, error: message }],
      metrics: { error: true, mode },
      output: { error: message },
    };
    state.lastReport = report;
    state.lastReportInfo = null;
    state.lastMetrics = report.metrics;
    state.lastDiagnosticsSuite = suite;
    updateDiagnosticsReport(timestamp);
    renderDiagnosticsOutput({ suite, modelId, report }, suite, captureOutput);
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
  const diagnosticsProfile = $('diagnostics-profile');
  const diagnosticsRun = $('diagnostics-run-btn');
  const diagnosticsVerify = $('diagnostics-verify-btn');
  const diagnosticsExport = $('diagnostics-export-btn');
  const unloadModelBtn = $('unload-model-btn');
  const clearMemoryBtn = $('clear-memory-btn');
  const modelsQuickModelsList = $('models-quick-models-list');
  const runPrompt = $('run-prompt');
  const runPromptShuffle = $('run-prompt-shuffle');
  const runGenerate = $('run-generate-btn');
  const runStop = $('run-stop-btn');
  const runClear = $('run-clear-btn');
  const pulseReset = $('pulse-reset-btn');
  const temperatureInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const maxTokensInput = $('max-tokens-input');
  const diffusionPrompt = $('diffusion-prompt');
  const diffusionPromptShuffle = $('diffusion-prompt-shuffle');
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

  const onQuickModelAction = (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest('button[data-quick-action][data-quick-model-id]');
    if (!(button instanceof HTMLButtonElement)) return;
    const action = button.dataset.quickAction || '';
    const modelId = button.dataset.quickModelId || '';
    if (!action || !modelId) return;
    runQuickModelAction(action, modelId).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      updateConvertStatus(`Quick model action failed: ${message}`, 0);
      updateDiagnosticsStatus(`Quick model action failed: ${message}`, true);
    });
  };

  modelsQuickModelsList?.addEventListener('click', onQuickModelAction);
  bindStarterPromptInput(runPrompt);
  bindStarterPromptInput(diffusionPrompt);
  bindStarterPromptInput(diffusionNegative);

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

  [
    'run',
    'diffusion',
    'energy',
    'diagnostics',
  ].forEach((scope) => {
    const button = $(`${scope}-empty-notice-btn`);
    button?.addEventListener('click', () => {
      handleEmptyNoticeAction(scope).catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        updateConvertStatus(`Quick model action failed: ${message}`, 0);
        updateDiagnosticsStatus(`Quick model action failed: ${message}`, true);
      });
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

  runtimePreset?.addEventListener('change', () => {
    const mode = state.uiMode;
    storeDiagnosticsSelection(mode, { preset: runtimePreset.value || DEFAULT_RUNTIME_PRESET, profile: '' });
    if (runtimePreset.value !== 'modes/debug') {
      clearDiagnosticsOutput();
    }
    applySelectedRuntimePreset();
  });

  diagnosticsModelSelect?.addEventListener('change', () => {
    selectDiagnosticsModel(diagnosticsModelSelect.value || null);
  });

  diagnosticsProfile?.addEventListener('change', () => {
    const selectedProfileId = diagnosticsProfile.value || '';
    const selectedProfile = decodeDiagnosticsProfileId(selectedProfileId);
    if (selectedProfile) {
      storeDiagnosticsSelection(state.uiMode, {
        profile: selectedProfileId,
        suite: selectedProfile.suite,
        preset: selectedProfile.preset,
      });
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

  runPromptShuffle?.addEventListener('click', () => {
    applyStarterPrompt(runPrompt, RUN_STARTER_PROMPTS, { force: true });
    if (state.uiMode === 'embedding') {
      refreshEmbeddingDemoDocuments({ force: true });
    }
    runPrompt?.focus();
    runPrompt?.select();
  });

  runStop?.addEventListener('click', () => {
    stopRunGeneration();
  });

  runClear?.addEventListener('click', () => {
    handleRunClear();
  });
  pulseReset?.addEventListener('click', () => {
    handleInferencePulseReset();
  });

  temperatureInput?.addEventListener('input', updateRunAutoLabels);
  topPInput?.addEventListener('input', updateRunAutoLabels);
  topKInput?.addEventListener('input', updateRunAutoLabels);
  maxTokensInput?.addEventListener('input', updateRunAutoLabels);
  diffusionPrompt?.addEventListener('input', updateDiffusionCharCounters);
  diffusionNegative?.addEventListener('input', updateDiffusionCharCounters);

  diffusionPromptShuffle?.addEventListener('click', () => {
    applyStarterPrompt(diffusionPrompt, DIFFUSION_STARTER_PROMPTS, { force: true });
    applyStarterPrompt(diffusionNegative, DIFFUSION_NEGATIVE_STARTER_PROMPTS, { force: true });
    updateDiffusionCharCounters();
    diffusionPrompt?.focus();
    diffusionPrompt?.select();
  });

  diffusionClear?.addEventListener('click', () => {
    if (diffusionPrompt) {
      diffusionPrompt.value = '';
      setStarterExampleInput(diffusionPrompt, false);
    }
    if (diffusionNegative) {
      diffusionNegative.value = '';
      setStarterExampleInput(diffusionNegative, false);
    }
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
  setStatusIndicator('Initializing...', 'info');
  ensurePrimaryModeControlStack();
  bindUI();
  prefillDemoTextInputs();
  updateDiffusionCharCounters();
  configureDownloadCallbacks({
    onModelRegistered: registerDownloadedModel,
    onModelsUpdated: refreshModelList,
    onProgress: handleDownloadProgressEvent,
    onStateChange: handleDownloadStateChangeEvent,
  });
  populateModelPresets();
  populateRuntimePresetSelects();
  populateEnergyDemoSelect();
  setUiMode(state.uiMode);
  await loadQuickModelCatalog();
  await refreshModelList();
  await refreshGpuInfo();
  await refreshDownloads();
  updateMemoryControls();
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
