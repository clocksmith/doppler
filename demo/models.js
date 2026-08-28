import { dr } from 'doppler-gpu/compat';
import { state } from './ui/state.js';
import { syncSendButton } from './input.js';
import { clearOutput } from './output.js';
import { setExportEnabled } from './report.js';

const LAST_USED_MODEL_STORAGE_KEY = 'doppler.demo.last-used-model';

let catalog = [];
let selectedModelId = null;
let onModelLoaded = null;
let onProgress = null;

function $(id) {
  return document.getElementById(id);
}

function modelLabel(entry) {
  return entry?.label || entry?.modelId || 'Unknown model';
}

function setStatus(text, busy = false) {
  const dot = $('status-dot');
  const label = $('status-text');
  dot?.classList.toggle('is-ready', !busy);
  dot?.classList.toggle('is-busy', busy);
  if (label) label.textContent = text;
}

function setProgress(event) {
  state.downloadProgress = event ?? null;
  const row = $('model-select-progress-row');
  const bar = $('model-select-progress');
  const fill = bar?.querySelector('.model-card-progress-fill');
  const label = $('model-select-progress-label');
  const percent = Math.max(0, Math.min(100, Number(event?.percent ?? 0)));
  if (row) row.hidden = !event;
  if (bar) bar.setAttribute('aria-valuenow', String(Math.round(percent)));
  if (fill) fill.style.width = `${percent}%`;
  if (label) label.textContent = `${Math.round(percent)}%`;
  onProgress?.(event);
}

function selectedEntry() {
  return catalog.find((entry) => entry.modelId === selectedModelId) ?? null;
}

function syncSelectedControls() {
  const entry = selectedEntry();
  const status = entry ? (state.modelStatus[entry.modelId] ?? 'available') : null;
  const action = $('model-select-action');
  const remove = $('model-select-remove');
  const detail = $('model-select-detail');
  if (action) {
    action.disabled = !entry || status === 'loading';
    action.textContent = status === 'loaded'
      ? 'Loaded'
      : (status === 'stored' ? 'Use saved model' : 'Load locally');
  }
  if (remove) {
    remove.hidden = status !== 'stored' && status !== 'loaded';
    remove.disabled = state.generating || status === 'loading';
  }
  if (detail) {
    detail.textContent = entry ? buildModelCardDetail(entry, status) : '';
  }
}

function createModelCard(entry) {
  const card = document.createElement('button');
  card.type = 'button';
  card.className = 'model-card';
  card.dataset.modelId = entry.modelId;
  card.textContent = modelLabel(entry);
  card.addEventListener('click', () => {
    selectedModelId = entry.modelId;
    const select = $('model-select');
    if (select) select.value = entry.modelId;
    syncSelectedControls();
  });
  return card;
}

async function loadSelectedModel() {
  const entry = selectedEntry();
  if (!entry || state.generating) return;
  state.modelStatus[entry.modelId] = 'loading';
  syncSelectedControls();
  setStatus('Loading model…', true);
  setProgress({ percent: 0, message: 'Preparing model' });
  try {
    const model = await dr.load(entry.modelId, {
      cache: 'opfs',
      onProgress: setProgress,
    });
    if (state.model && state.model !== model) {
      await state.model.unload().catch(() => {});
    }
    for (const modelId of Object.keys(state.modelStatus)) {
      if (state.modelStatus[modelId] === 'loaded') state.modelStatus[modelId] = 'stored';
    }
    state.model = model;
    state.modelId = model.modelId;
    state.modelStatus[entry.modelId] = 'loaded';
    localStorage.setItem(LAST_USED_MODEL_STORAGE_KEY, entry.modelId);
    setStatus('Ready');
    onModelLoaded?.(model, entry.modelId);
  } catch (error) {
    state.modelStatus[entry.modelId] = 'available';
    setStatus('Load failed');
    throw error;
  } finally {
    setProgress(null);
    renderModelCards();
  }
}

async function removeSelectedModel() {
  const entry = selectedEntry();
  if (!entry || state.generating) return;
  const removed = await dr.removePersistentModel(entry.modelId);
  if (!removed) return;
  if (state.modelId === entry.modelId) {
    state.model = null;
    state.modelId = null;
    state.lastRun = null;
    clearOutput();
    setExportEnabled(false);
    syncSendButton();
  }
  state.modelStatus[entry.modelId] = 'available';
  localStorage.removeItem(LAST_USED_MODEL_STORAGE_KEY);
  renderModelCards();
}

function bindModelControls() {
  const select = $('model-select');
  if (select && select.dataset.bound !== 'true') {
    select.dataset.bound = 'true';
    select.addEventListener('change', () => {
      selectedModelId = select.value || null;
      syncSelectedControls();
    });
  }
  const action = $('model-select-action');
  if (action && action.dataset.bound !== 'true') {
    action.dataset.bound = 'true';
    action.addEventListener('click', () => {
      loadSelectedModel().catch((error) => setStatus(`Load failed: ${error.message}`));
    });
  }
  const remove = $('model-select-remove');
  if (remove && remove.dataset.bound !== 'true') {
    remove.dataset.bound = 'true';
    remove.addEventListener('click', () => {
      removeSelectedModel().catch((error) => setStatus(`Remove failed: ${error.message}`));
    });
  }
}

export function canRemoveModelStatus(status) {
  return status === 'stored' || status === 'loaded';
}

export function buildModelCardDetail(entry, status) {
  const label = status === 'loaded'
    ? 'Loaded'
    : (status === 'stored' ? 'Downloaded' : 'Available');
  return `${label} · ${entry?.modelId ?? 'unknown'}`;
}

export function setModelCallbacks(callbacks = {}) {
  onModelLoaded = callbacks.onLoaded ?? null;
  onProgress = callbacks.onDownloadProgress ?? null;
}

export async function loadCatalog() {
  const entries = await dr.listModelDetails();
  catalog = entries.map((entry) => ({ ...entry }));
  selectedModelId = catalog[0]?.modelId ?? null;
  state.quickModelCatalog = catalog.map((entry) => ({ ...entry }));
  bindModelControls();
  renderModelCards();
  return catalog.map((entry) => ({ ...entry }));
}

export function selectDefaultStoredModel(catalogEntries, registeredEntries, preferredModelId = null) {
  const stored = new Set(
    (Array.isArray(registeredEntries) ? registeredEntries : [])
      .map((entry) => entry?.modelId)
      .filter(Boolean)
  );
  if (preferredModelId && stored.has(preferredModelId)) {
    return catalogEntries.find((entry) => entry.modelId === preferredModelId) ?? null;
  }
  return catalogEntries.find((entry) => stored.has(entry.modelId)) ?? null;
}

export async function checkStoredModels() {
  const registered = await dr.listPersistentModels();
  const stored = new Set(registered.map((entry) => entry.modelId));
  for (const entry of catalog) {
    state.modelStatus[entry.modelId] = stored.has(entry.modelId) ? 'stored' : 'available';
  }
  renderModelCards();
  return registered;
}

export async function loadDefaultStoredModel() {
  const registered = await dr.listPersistentModels();
  const preferred = localStorage.getItem(LAST_USED_MODEL_STORAGE_KEY);
  const entry = selectDefaultStoredModel(catalog, registered, preferred);
  if (!entry) return null;
  selectedModelId = entry.modelId;
  renderModelCards();
  await loadSelectedModel();
  return state.model;
}

export function renderModelCards() {
  const select = $('model-select');
  if (select) {
    select.replaceChildren();
    for (const entry of catalog) {
      const option = document.createElement('option');
      option.value = entry.modelId;
      option.textContent = modelLabel(entry);
      select.append(option);
    }
    select.disabled = catalog.length === 0;
    select.value = selectedModelId ?? '';
  }
  const cards = $('model-cards');
  if (cards) {
    cards.replaceChildren(...catalog.map(createModelCard));
  }
  const count = $('model-browser-count');
  if (count) count.textContent = `${catalog.length} models`;
  const status = $('model-select-status');
  if (status) status.textContent = catalog.length
    ? `${catalog.length} supported`
    : 'No supported models';
  syncSelectedControls();
}
