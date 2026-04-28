import {
  openModelStore,
  parseManifest,
  saveManifest,
  saveTensorsToStore,
  saveTokenizer,
  saveTokenizerModel,
  writeShard,
  loadManifestFromStore,
  loadTensorsFromStore,
  loadTokenizerFromStore,
  loadTokenizerModelFromStore,
  listRegisteredModels,
  registerModel,
  removeRegisteredModel,
  deleteModel,
  initDevice,
  formatBytes,
  DEFAULT_MANIFEST_INFERENCE,
} from 'doppler-gpu/tooling';
import { createPipeline } from 'doppler-gpu/generation';
import {
  DEFAULT_EXECUTION_V1_SESSION,
  EXECUTION_V1_SCHEMA_ID,
} from '../src/config/schema/index.js';
import { state } from './ui/state.js';
import { setRunEnabled } from './input.js';
import { clearOutput } from './output.js';
import { setExportEnabled } from './report.js';

const HF_RESOLVE_BASE = 'https://huggingface.co';
const CATALOG_URL = new URL('../models/catalog.json', import.meta.url).toString();

let catalog = [];
let onModelLoaded = null;
let onProgress = null;

function $(id) { return document.getElementById(id); }
function isPlainObject(value) { return value != null && typeof value === 'object' && !Array.isArray(value); }

function normalizeBaseUrl(value) {
  return typeof value === 'string' && value.trim().length > 0
    ? value.trim().replace(/\/$/, '')
    : null;
}

function hfUrl(repoId, revision, path, file) {
  return `${HF_RESOLVE_BASE}/${repoId}/resolve/${revision}/${path}/${file}`;
}

function sizeLabel(bytes) {
  if (!bytes) return '';
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
}

function isDemoVisibleEntry(entry) {
  if (entry?.demoVisible === false) {
    return false;
  }
  return entry?.quickstart === true || entry?.demoVisible === true;
}

function getDemoWarningBadges(entry) {
  return Array.isArray(entry?.demoWarningBadges)
    ? entry.demoWarningBadges
      .map((badge) => (typeof badge === 'string' ? badge.trim() : ''))
      .filter((badge) => badge.length > 0)
    : [];
}

function getDemoWarningText(entry) {
  return typeof entry?.demoWarningText === 'string' ? entry.demoWarningText.trim() : '';
}

export function canRemoveModelStatus(status) {
  return status === 'stored' || status === 'loaded';
}

export function buildRemoveConfirmText(entry) {
  const sizeTxt = sizeLabel(entry?.sizeBytes);
  return sizeTxt ? `Remove ${sizeTxt} from OPFS?` : 'Remove this model from OPFS?';
}

export function buildModelCardDetail(entry, status) {
  const sizeTxt = sizeLabel(entry?.sizeBytes);
  const STATUS_LABELS = {
    loaded: 'Active',
    loading: 'Loading to GPU...',
    stored: 'Ready',
    downloading: 'Downloading...',
  };
  const statusLabel = STATUS_LABELS[status];
  if (!statusLabel) {
    return sizeTxt;
  }
  if (!sizeTxt || status === 'loading' || status === 'downloading') {
    return statusLabel;
  }
  return `${statusLabel} · ${sizeTxt}`;
}

function setIdleStatus(text = 'Select model') {
  const dot = $('status-dot');
  const statusText = $('status-text');
  if (dot) {
    dot.classList.remove('is-ready', 'is-busy');
  }
  if (statusText) {
    statusText.textContent = text;
  }
}

function clearLoadedModelState() {
  state.modelId = null;
  state.pipeline = null;
  state.lastRun = null;
  state.downloadProgress = null;
  setRunEnabled(false);
  setExportEnabled(false);
  clearOutput();
  setIdleStatus();
}

async function confirmRemoveModel(entry) {
  const message = buildRemoveConfirmText(entry);
  const detailText = entry?.label
    ? `${entry.label} will be removed from OPFS and will need to be downloaded again.`
    : 'This model will be removed from OPFS and will need to be downloaded again.';
  const dialog = $('remove-model-dialog');
  const messageEl = $('remove-model-message');
  const detailEl = $('remove-model-detail');

  if (!dialog || typeof dialog.showModal !== 'function') {
    return typeof window === 'object' && typeof window.confirm === 'function'
      ? window.confirm(message)
      : false;
  }

  if (messageEl) {
    messageEl.textContent = message;
  }
  if (detailEl) {
    detailEl.textContent = detailText;
  }

  if (dialog.open) {
    dialog.close('cancel');
  }

  return new Promise((resolve) => {
    const handleClose = () => {
      resolve(dialog.returnValue === 'confirm');
    };
    dialog.addEventListener('close', handleClose, { once: true });
    dialog.showModal();
  });
}

async function removeStoredModel(entry) {
  if (!entry?.modelId || state.generating || state.prefilling) {
    return;
  }
  const confirmed = await confirmRemoveModel(entry);
  if (!confirmed) {
    return;
  }

  const removed = await deleteModel(entry.modelId);
  if (!removed) {
    throw new Error(`Could not remove ${entry.modelId} from OPFS.`);
  }
  await removeRegisteredModel(entry.modelId);
  if (state.modelId === entry.modelId || state.modelStatus[entry.modelId] === 'loaded') {
    clearLoadedModelState();
  }
  state.modelStatus[entry.modelId] = 'available';
  renderModelCards();
}

export function setModelCallbacks({ onLoaded, onDownloadProgress }) {
  onModelLoaded = onLoaded ?? null;
  onProgress = onDownloadProgress ?? null;
}

export function buildLocalModelBaseUrl(modelId, origin = null) {
  const normalizedOrigin = typeof origin === 'string' && origin.trim().length > 0 ? origin.trim() : null;
  if (!normalizedOrigin || typeof modelId !== 'string' || modelId.trim().length === 0) {
    return null;
  }
  return new URL(`/models/local/${encodeURIComponent(modelId.trim())}`, normalizedOrigin).toString().replace(/\/$/, '');
}

function buildHfModelBaseUrl(entry) {
  const repoId = entry?.hf?.repoId;
  const revision = entry?.hf?.revision;
  const repoPath = entry?.hf?.path;
  if (!repoId || !revision || !repoPath) {
    return null;
  }
  return hfUrl(repoId, revision, repoPath, '').replace(/\/$/, '');
}

function buildArtifactUrl(baseUrl, path) {
  return new URL(path, `${baseUrl.replace(/\/$/, '')}/`).toString();
}

async function probeManifestBaseUrl(baseUrl, fetchImpl = fetch) {
  if (!baseUrl) {
    return false;
  }
  const probeUrl = `${buildArtifactUrl(baseUrl, 'manifest.json')}?probe=${Date.now()}`;
  try {
    const head = await fetchImpl(probeUrl, {
      method: 'HEAD',
      cache: 'no-store',
    });
    if (head.ok) {
      return true;
    }
    if (head.status !== 405) {
      return false;
    }
  } catch {
    return false;
  }
  try {
    const get = await fetchImpl(probeUrl, {
      cache: 'no-store',
    });
    return get.ok;
  } catch {
    return false;
  }
}

async function resolveLocalCatalogSourceMap(entries, fetchImpl = fetch, origin = null) {
  const localBaseUrls = new Map();
  await Promise.all(entries.map(async (entry) => {
    const localBaseUrl = buildLocalModelBaseUrl(entry?.modelId, origin);
    if (!localBaseUrl) {
      return;
    }
    if (await probeManifestBaseUrl(localBaseUrl, fetchImpl)) {
      localBaseUrls.set(entry.modelId, localBaseUrl);
    }
  }));
  return localBaseUrls;
}

export function selectDemoCatalogEntries(models, options = {}) {
  const entries = Array.isArray(models) ? models : [];
  const localBaseUrls = options.localBaseUrls instanceof Map ? options.localBaseUrls : new Map();
  const selected = entries.filter((entry) => {
    if (!entry?.modes?.includes('text')) {
      return false;
    }
    if (!isDemoVisibleEntry(entry)) {
      return false;
    }
    if (entry?.artifactCompleteness !== 'complete') {
      return false;
    }
    if (entry?.runtimePromotionState !== 'manifest-owned') {
      return false;
    }
    if (entry?.weightsRefAllowed !== false) {
      return false;
    }
    if (localBaseUrls.has(entry.modelId)) {
      return true;
    }
    if (normalizeBaseUrl(entry?.baseUrl)) {
      return true;
    }
    try {
      return normalizeBaseUrl(buildHfModelBaseUrl(entry)) != null;
    } catch {
      return false;
    }
  }).map((entry) => ({
    ...entry,
    localBaseUrl: localBaseUrls.get(entry.modelId) ?? null,
  }));
  selected.sort((a, b) => {
    if (a.recommended !== b.recommended) {
      return a.recommended ? -1 : 1;
    }
    if ((a.sortOrder ?? 999) !== (b.sortOrder ?? 999)) {
      return (a.sortOrder ?? 999) - (b.sortOrder ?? 999);
    }
    return String(a.label || a.modelId || '').localeCompare(String(b.label || b.modelId || ''));
  });
  return selected;
}

export function buildModelSourceCandidates(entry) {
  const candidates = [];
  const seen = new Set();
  const pushCandidate = (kind, baseUrl) => {
    const normalized = normalizeBaseUrl(baseUrl);
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    candidates.push({ kind, baseUrl: normalized });
  };

  pushCandidate('local', entry?.localBaseUrl);
  pushCandidate('catalog', entry?.baseUrl);
  pushCandidate('hf', buildHfModelBaseUrl(entry));
  return candidates;
}

async function resolveManifestSource(entry, signal) {
  const candidates = buildModelSourceCandidates(entry);
  if (candidates.length === 0) {
    throw new Error(`Model "${entry?.modelId ?? 'unknown'}" does not have a downloadable source.`);
  }

  const errors = [];
  for (const candidate of candidates) {
    const manifestUrl = buildArtifactUrl(candidate.baseUrl, 'manifest.json');
    try {
      const manifestText = await fetchText(manifestUrl, signal);
      const manifest = parseManifest(manifestText);
      const missingArtifacts = await validateManifestArtifacts(candidate.baseUrl, manifest, signal);
      if (missingArtifacts.length > 0) {
        errors.push(`${candidate.kind}: missing files [${missingArtifacts.join(', ')}]`);
        continue;
      }
      return {
        kind: candidate.kind,
        baseUrl: candidate.baseUrl,
        manifestText,
        manifest,
      };
    } catch (error) {
      errors.push(`${candidate.kind}: ${error.message}`);
    }
  }

  throw new Error(
    `Could not fetch manifest for "${entry?.modelId ?? 'unknown'}" from any configured source. ` +
    errors.join(' | ')
  );
}

async function validateManifestArtifacts(baseUrl, manifest, signal) {
  const requiredFiles = [];
  if (typeof manifest?.tokenizer?.file === 'string' && manifest.tokenizer.file.trim().length > 0) {
    requiredFiles.push(manifest.tokenizer.file.trim());
  }
  if (
    typeof manifest?.tokenizer?.sentencepieceModel === 'string'
    && manifest.tokenizer.sentencepieceModel.trim().length > 0
  ) {
    requiredFiles.push(manifest.tokenizer.sentencepieceModel.trim());
  }
  if (typeof manifest?.tensorsFile === 'string' && manifest.tensorsFile.trim().length > 0) {
    requiredFiles.push(manifest.tensorsFile.trim());
  }
  if (Array.isArray(manifest?.shards)) {
    for (const shard of manifest.shards) {
      if (typeof shard?.filename === 'string' && shard.filename.trim().length > 0) {
        requiredFiles.push(shard.filename.trim());
      }
    }
  }

  if (requiredFiles.length === 0) {
    return [];
  }

  const checks = await Promise.all(requiredFiles.map(async (file) => ({
    file,
    present: await probeManifestAsset(baseUrl, file, signal),
  })));
  return checks.filter((entry) => !entry.present).map((entry) => entry.file);
}

async function probeManifestAsset(baseUrl, path, signal, fetchImpl = fetch) {
  if (!baseUrl || typeof path !== 'string' || path.trim().length === 0) {
    return false;
  }
  const url = `${buildArtifactUrl(baseUrl, path.trim())}?probe=${Date.now()}`;
  try {
    const head = await fetchImpl(url, {
      method: 'HEAD',
      signal,
      cache: 'no-store',
    });
    if (head.ok) {
      return true;
    }
    if (head.status !== 405) {
      return false;
    }
  } catch {
    return false;
  }
  try {
    const get = await fetchImpl(url, { signal, cache: 'no-store' });
    return get.ok;
  } catch {
    return false;
  }
}

export async function loadCatalog() {
  try {
    const res = await fetch(`${CATALOG_URL}?t=${Date.now()}`);
    const data = await res.json();
    const models = Array.isArray(data?.models) ? data.models : [];
    const origin = typeof window === 'object' && window.location?.origin
      ? window.location.origin
      : null;
    const localBaseUrls = await resolveLocalCatalogSourceMap(models, fetch, origin);
    catalog = selectDemoCatalogEntries(models, { localBaseUrls });
  } catch {
    catalog = [];
  }
}

export async function checkStoredModels() {
  try {
    const registered = await listRegisteredModels();
    const storedIds = new Set(registered.map((r) => r.modelId));
    for (const entry of catalog) {
      if (storedIds.has(entry.modelId)) {
        state.modelStatus[entry.modelId] = 'stored';
      } else {
        state.modelStatus[entry.modelId] = 'available';
      }
    }
  } catch {
    // OPFS unavailable — mark all as available for download
    for (const entry of catalog) {
      state.modelStatus[entry.modelId] = 'available';
    }
  }
}

export function renderModelCards() {
  const container = $('model-cards');
  if (!container) return;
  container.innerHTML = '';

  for (const entry of catalog) {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.dataset.modelId = entry.modelId;

    const status = state.modelStatus[entry.modelId] || 'available';
    if (status === 'loaded') card.classList.add('is-active');
    else if (status === 'loading') card.classList.add('is-loading');
    else if (status === 'stored') card.classList.add('is-stored');
    else if (status === 'downloading') card.classList.add('is-downloading');

    const top = document.createElement('div');
    top.className = 'model-card-top';

    const copy = document.createElement('div');
    copy.className = 'model-card-copy';

    const name = document.createElement('div');
    name.className = 'model-card-name';
    name.textContent = entry.label;

    const detail = document.createElement('div');
    detail.className = 'model-card-detail';
    detail.textContent = buildModelCardDetail(entry, status);

    copy.appendChild(name);
    copy.appendChild(detail);

    const warningBadges = getDemoWarningBadges(entry);
    if (warningBadges.length > 0) {
      const badges = document.createElement('div');
      badges.className = 'model-card-badges';
      for (const warningBadge of warningBadges) {
        const badge = document.createElement('span');
        badge.className = 'model-card-badge';
        badge.textContent = warningBadge;
        badges.appendChild(badge);
      }
      copy.appendChild(badges);
    }

    const warningText = getDemoWarningText(entry);
    if (warningText) {
      const warning = document.createElement('div');
      warning.className = 'model-card-warning';
      warning.textContent = warningText;
      copy.appendChild(warning);
      card.title = warningText;
    }
    top.appendChild(copy);

    if (canRemoveModelStatus(status)) {
      const removeButton = document.createElement('button');
      removeButton.type = 'button';
      removeButton.className = 'btn btn-ghost model-card-remove';
      removeButton.textContent = 'Remove';
      removeButton.setAttribute('aria-label', `Remove ${entry.label} from OPFS`);
      removeButton.addEventListener('click', async (event) => {
        event.preventDefault();
        event.stopPropagation();
        try {
          await removeStoredModel(entry);
        } catch (error) {
          console.error(`Could not remove ${entry.modelId} from OPFS`, error);
        }
      });
      top.appendChild(removeButton);
    }

    card.appendChild(top);

    if (status === 'downloading') {
      const bar = document.createElement('div');
      bar.className = 'model-card-progress';
      bar.id = `progress-${entry.modelId}`;
      bar.style.width = '0%';
      card.appendChild(bar);
    }

    card.addEventListener('click', () => {
      void handleCardClick(entry).catch((error) => {
        console.error(`Failed to load model ${entry.modelId}:`, error);
      });
    });
    container.appendChild(card);
  }
}

let loadingLock = false;

async function handleCardClick(entry) {
  if (loadingLock) return;
  const status = state.modelStatus[entry.modelId];
  if (status === 'downloading' || status === 'loaded') return;

  loadingLock = true;
  try {
    if (status === 'stored') {
      await loadModelFromStorage(entry.modelId);
    } else {
      await downloadAndLoad(entry);
    }
  } finally {
    loadingLock = false;
  }
}

async function downloadAndLoad(entry) {
  const { modelId } = entry;

  state.modelStatus[modelId] = 'downloading';
  renderModelCards();

  const abort = new AbortController();
  const signal = abort.signal;

  try {
    const resolvedSource = await resolveManifestSource(entry, signal);

    // Fetch manifest and patch compat fields before storing
    const manifest = patchManifestCompat(resolvedSource.manifest ?? parseManifest(resolvedSource.manifestText));
    manifest.modelId = modelId;

    await openModelStore(modelId);
    await saveManifest(JSON.stringify(manifest, null, 2));

    // Tokenizer
    const tokFile = manifest?.tokenizer?.file;
    if (tokFile) {
      if (tokFile.endsWith('.model')) {
        const bytes = await fetchBytes(buildArtifactUrl(resolvedSource.baseUrl, tokFile), signal);
        await saveTokenizerModel(bytes.buffer);
      } else {
        const text = await fetchText(buildArtifactUrl(resolvedSource.baseUrl, tokFile), signal);
        await saveTokenizer(text);
      }
    }

    // Tensors map
    if (manifest.tensorsFile) {
      const text = await fetchText(buildArtifactUrl(resolvedSource.baseUrl, manifest.tensorsFile), signal);
      await saveTensorsToStore(text);
    }

    // Shards
    const shards = manifest.shards || [];
    for (let i = 0; i < shards.length; i++) {
      const shard = shards[i];
      const bytes = await fetchBytes(buildArtifactUrl(resolvedSource.baseUrl, shard.filename), signal);
      await writeShard(shard.index, bytes, { verify: true });

      const percent = ((i + 1) / shards.length) * 100;
      state.downloadProgress = { percent, currentShard: i + 1, totalShards: shards.length };
      updateProgressBar(modelId, percent);
      if (onProgress) onProgress({ modelId, percent, currentShard: i + 1, totalShards: shards.length });
    }

    await registerModel({ modelId });
    state.modelStatus[modelId] = 'stored';
    state.downloadProgress = null;
    renderModelCards();

    // Auto-load after download
    await loadModelFromStorage(modelId);
  } catch (err) {
    state.modelStatus[modelId] = 'available';
    state.downloadProgress = null;
    renderModelCards();
    throw err;
  }
}

export function patchManifestCompat(manifest) {
  // Fill missing nullable-required inference fields with schema defaults
  // so older HF manifests pass validation without re-conversion.
  const defaults = DEFAULT_MANIFEST_INFERENCE;
  if (!manifest.inference) manifest.inference = {};
  const inf = manifest.inference;

  const fillMissingFields = (target, source, options = {}) => {
    const treatNullAsMissing = options.treatNullAsMissing === true;
    if (!isPlainObject(source)) {
      return target;
    }
    if (!isPlainObject(target)) {
      target = {};
    }
    for (const [key, defaultValue] of Object.entries(source)) {
      const currentValue = target[key];
      const missing = currentValue === undefined || (treatNullAsMissing && currentValue === null);
      if (isPlainObject(defaultValue)) {
        if (missing || !isPlainObject(currentValue)) {
          target[key] = fillMissingFields({}, defaultValue, options);
          continue;
        }
        target[key] = fillMissingFields(currentValue, defaultValue, options);
        continue;
      }
      if (missing) {
        target[key] = defaultValue;
      }
    }
    return target;
  };

  for (const section of ['attention', 'normalization', 'ffn', 'rope', 'output', 'layerPattern', 'chatTemplate']) {
    inf[section] = fillMissingFields(inf[section], defaults[section]);
  }

  if (inf.schema === EXECUTION_V1_SCHEMA_ID) {
    inf.session = fillMissingFields(inf.session, DEFAULT_EXECUTION_V1_SESSION, { treatNullAsMissing: true });
  }
  return manifest;
}

async function loadModelFromStorage(modelId) {
  state.modelStatus[modelId] = 'loading';
  renderModelCards();

  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) throw new Error(`No manifest for ${modelId}`);

  const manifest = patchManifestCompat(parseManifest(manifestText));
  // Re-save patched manifest so the loader reads compat fields from OPFS
  await saveManifest(JSON.stringify(manifest, null, 2));
  await initDevice();

  const tensorsText = await loadTensorsFromStore();
  const tokenizerText = await loadTokenizerFromStore();
  const tokenizerModel = await loadTokenizerModelFromStore();

  const pipeline = await createPipeline(manifest, {
    tensorsJson: tensorsText ?? undefined,
    tokenizerJson: tokenizerText ?? undefined,
    tokenizerModel: tokenizerModel ?? undefined,
  });

  // Mark all models as not-loaded, then mark this one
  for (const id of Object.keys(state.modelStatus)) {
    if (state.modelStatus[id] === 'loaded') state.modelStatus[id] = 'stored';
  }
  state.modelId = modelId;
  state.modelStatus[modelId] = 'loaded';
  state.pipeline = pipeline;
  renderModelCards();

  if (onModelLoaded) onModelLoaded(pipeline, modelId);
}

function updateProgressBar(modelId, percent) {
  const bar = $(`progress-${modelId}`);
  if (bar) bar.style.width = `${Math.min(100, percent)}%`;
}

async function fetchText(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return res.text();
}

async function fetchBytes(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}
