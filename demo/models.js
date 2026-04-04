import {
  openModelStore,
  parseManifest,
  saveManifest,
  saveTensorsToStore,
  saveTokenizer,
  saveTokenizerModel,
  saveAuxFile,
  writeShard,
  loadManifestFromStore,
  loadTensorsFromStore,
  loadTokenizerFromStore,
  loadTokenizerModelFromStore,
  listRegisteredModels,
  registerModel,
  createPipeline,
  initDevice,
  formatBytes,
  DEFAULT_MANIFEST_INFERENCE,
} from 'doppler-gpu';
import { state } from './ui/state.js';

const HF_RESOLVE_BASE = 'https://huggingface.co';
const AUX_FILENAMES = ['config.json', 'generation_config.json', 'tokenizer_config.json', 'special_tokens_map.json'];
const CATALOG_URL = typeof window === 'object' && window.location?.origin
  ? new URL('/models/catalog.json', window.location.origin).toString()
  : new URL('../models/catalog.json', import.meta.url).toString();

let catalog = [];
let onModelLoaded = null;
let onProgress = null;

function $(id) { return document.getElementById(id); }

function hfUrl(repoId, revision, path, file) {
  return `${HF_RESOLVE_BASE}/${repoId}/resolve/${revision}/${path}/${file}`;
}

function sizeLabel(bytes) {
  if (!bytes) return '';
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
}

export function setModelCallbacks({ onLoaded, onDownloadProgress }) {
  onModelLoaded = onLoaded ?? null;
  onProgress = onDownloadProgress ?? null;
}

export async function loadCatalog() {
  try {
    const res = await fetch(`${CATALOG_URL}?t=${Date.now()}`);
    const data = await res.json();
    catalog = (data.models || []).filter(
      (m) => m.quickstart && m.modes?.includes('text')
    );
    catalog.sort((a, b) => (a.sortOrder ?? 999) - (b.sortOrder ?? 999));
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

    const name = document.createElement('div');
    name.className = 'model-card-name';
    name.textContent = entry.label;

    const detail = document.createElement('div');
    detail.className = 'model-card-detail';
    const sizeTxt = sizeLabel(entry.sizeBytes);
    const STATUS_LABELS = {
      loaded: 'Active',
      loading: 'Loading to GPU...',
      stored: 'Ready',
      downloading: 'Downloading...',
    };
    detail.textContent = STATUS_LABELS[status] || sizeTxt;

    card.appendChild(name);
    card.appendChild(detail);

    if (status === 'downloading') {
      const bar = document.createElement('div');
      bar.className = 'model-card-progress';
      bar.id = `progress-${entry.modelId}`;
      bar.style.width = '0%';
      card.appendChild(bar);
    }

    card.addEventListener('click', () => handleCardClick(entry));
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
  const { modelId, hf } = entry;
  if (!hf) return;

  state.modelStatus[modelId] = 'downloading';
  renderModelCards();

  const baseUrl = (path) => hfUrl(hf.repoId, hf.revision, hf.path, path);
  const abort = new AbortController();
  const signal = abort.signal;

  try {
    // Fetch manifest and patch compat fields before storing
    const manifestText = await fetchText(baseUrl('manifest.json'), signal);
    const manifest = patchManifestCompat(parseManifest(manifestText));
    manifest.modelId = modelId;

    await openModelStore(modelId);
    await saveManifest(JSON.stringify(manifest, null, 2));

    // Tokenizer
    const tokFile = manifest?.tokenizer?.file;
    if (tokFile) {
      if (tokFile.endsWith('.model')) {
        const bytes = await fetchBytes(baseUrl(tokFile), signal);
        await saveTokenizerModel(bytes.buffer);
      } else {
        const text = await fetchText(baseUrl(tokFile), signal);
        await saveTokenizer(text);
      }
    }

    // Tensors map
    if (manifest.tensorsFile) {
      const text = await fetchText(baseUrl(manifest.tensorsFile), signal);
      await saveTensorsToStore(text);
    }

    // Aux files
    for (const name of AUX_FILENAMES) {
      try {
        const bytes = await fetchBytes(baseUrl(name), signal);
        await saveAuxFile(name, bytes.buffer);
      } catch (e) {
        if (!String(e?.message).includes('404')) throw e;
      }
    }

    // Shards
    const shards = manifest.shards || [];
    for (let i = 0; i < shards.length; i++) {
      const shard = shards[i];
      const bytes = await fetchBytes(baseUrl(shard.filename), signal);
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

function patchManifestCompat(manifest) {
  // Fill missing nullable-required inference fields with schema defaults
  // so older HF manifests pass validation without re-conversion.
  const defaults = DEFAULT_MANIFEST_INFERENCE;
  if (!manifest.inference) manifest.inference = {};
  const inf = manifest.inference;
  for (const section of ['attention', 'normalization', 'ffn', 'rope', 'output', 'layerPattern', 'chatTemplate']) {
    if (!inf[section]) inf[section] = {};
    const sectionDefaults = defaults[section];
    if (!sectionDefaults || typeof sectionDefaults !== 'object') continue;
    for (const [key, defaultValue] of Object.entries(sectionDefaults)) {
      if (inf[section][key] === undefined) {
        inf[section][key] = defaultValue;
      }
    }
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
