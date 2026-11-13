import { dr } from 'doppler-gpu/compat';
import { state } from './ui/state.js';
import { clearConversationHistory, syncSendButton } from './input.js';
import { setExportEnabled } from './report.js';
import { setPhase } from './output.js';

export const DEFAULT_DEMO_MODEL_ID = 'gemma-3-270m-it-q4k-ehf16-af32';

const LAST_USED_MODEL_STORAGE_KEY = 'doppler.demo.last-used-model';

let catalog = [];
let selectedModelId = null;
let onModelLoaded = null;
let onProgress = null;

function $(id) {
  return document.getElementById(id);
}

// The published catalog label, including its configuration, e.g. "(Q4K/F32a)".
export function modelLabel(entry) {
  return entry?.label || entry?.modelId || 'Unknown model';
}

function downloadSize(entry) {
  const bytes = entry?.sizeBytes;
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.ceil(bytes / 1e6)} MB`;
}

// Shown beside Load when the browser cannot start WebGPU at all.
export const NO_WEBGPU_NOTE = 'This browser can’t run models here: WebGPU isn’t available. The recorded answer below still works. '
  + 'Reloading, updating the browser or turning on hardware acceleration sometimes helps.';

// Asks for an adapter only. Its limits are per-buffer GPU limits, not memory.
export async function probeWebGpu(gpu = globalThis.navigator?.gpu) {
  if (!gpu) return { available: false, limits: null };
  // An object that cannot be asked gives no verdict here; loading a model will find out.
  if (typeof gpu.requestAdapter !== 'function') return { available: true, limits: null };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { available: false, limits: null };
    return { available: true, limits: {
      maxBufferSize: adapter.limits?.maxBufferSize ?? null,
      maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize ?? null,
    } };
  } catch {
    return { available: false, limits: null };
  }
}

export async function checkWebGpu() {
  const result = await probeWebGpu();
  state.webgpuAvailable = result.available;
  state.gpuLimits = result.limits;
  syncModelControls();
  return result.available;
}

function isNoAdapterError(error) {
  return /DOPPLER_GPU_UNSUPPORTED_ADAPTER|DOPPLER_GPU_UNAVAILABLE/.test(String(error?.code ?? ''))
    || /Failed to get WebGPU adapter|WebGPU is not available/i.test(String(error?.message ?? error));
}

// Beside Load after a failure. `facts` holds only what was observed about model files.
export function loadFailureNote(entry, error, facts = {}) {
  const message = String(error?.message ?? error).replace(/\.$/, '');
  const files = facts.wasStored ? ' The copy saved in this browser was kept.'
    : facts.storedNow ? ' Its files finished downloading and are saved in this browser, so the next try will not download them again.'
    : facts.sawModelFiles ? ' Some of its files may have been downloaded before loading stopped; the browser keeps any that finished.'
    : ' No model weights were downloaded.';
  if (isNoAdapterError(error)) return `${NO_WEBGPU_NOTE}${files}`;
  if (/binding|maxStorageBufferBindingSize|maxBufferSize/i.test(message)) {
    return `${modelLabel(entry)} needs a single GPU buffer larger than this device allows. That is a per-buffer GPU limit, separate from how much memory the device has. ${message}.${files}`;
  }
  return `${modelLabel(entry)} could not load: ${message}.${files}`;
}

// In words, what Send would do about the model, or why it cannot.
export function sendGuidance({ hasPrompt = false } = {}) {
  if (!state.webgpuAvailable) return 'Live inference is not available in this browser. The recorded run still works.';
  if (state.model) return hasPrompt ? '' : 'Type a message to send.';
  const entry = selectedEntry();
  if (!entry) return 'Choose a model above first.';
  const status = state.modelStatus[entry.modelId] ?? 'available';
  if (status === 'loading' || state.modelBusy) return '';
  if (modelFit(entry) === 'too-large') {
    return `Send is off for ${modelLabel(entry)}: it needs more memory than this device reports. Choose a smaller model, or press Load anyway.`;
  }
  const size = downloadSize(entry);
  const first = status === 'stored' ? `load ${modelLabel(entry)} from the copy saved in this browser`
    : `download${size ? ` about ${size}` : ''} and load ${modelLabel(entry)}`;
  return hasPrompt ? `Send will first ${first}.` : `Type a message. Sending it will first ${first}.`;
}

// No WebGPU, or the chosen model looks too large to load unasked.
export function sendBlocked() {
  return !state.webgpuAvailable || (!state.model && modelFit(selectedEntry()) === 'too-large');
}

const GIB = 2 ** 30;

// The only memory figure a page can read: system memory in GiB, rounded down
// and capped at 8 by the browsers that expose it. WebGPU reports no GPU memory.
function reportedMemoryBytes() {
  const gib = globalThis.navigator?.deviceMemory;
  return Number.isFinite(gib) && gib > 0 ? gib * GIB : null;
}

function memorySize(bytes) {
  return `${Number((bytes / GIB).toFixed(1))} GB`;
}

// The catalog's stated GPU memory need against what the device reports, before
// any download. 'unknown' without both figures, or at the 8 GiB reporting cap.
export function modelFit(entry, reportedBytes = reportedMemoryBytes()) {
  const required = entry?.quickstartRequirements?.vramRequiredBytes;
  if (!Number.isFinite(required) || required <= 0 || reportedBytes == null) return 'unknown';
  if (required <= reportedBytes) return 'fits';
  return reportedBytes >= 8 * GIB ? 'unknown' : 'too-large';
}

export function isChatModel(entry) {
  return entry?.modes?.includes('text') === true && entry?.classification?.tasks?.includes('generation') === true;
}

function setProgress(event) {
  state.downloadProgress = event ?? null;
  if (event) setPhase(event.message?.trim() || 'Loading model…', { busy: true, quiet: true });
  onProgress?.(event);
}

function selectedEntry() {
  return catalog.find((entry) => entry.modelId === selectedModelId) ?? null;
}

export function syncModelControls() {
  const entry = selectedEntry();
  const status = entry ? (state.modelStatus[entry.modelId] ?? 'available') : null;
  const action = $('model-select-action');
  const remove = $('model-select-remove');
  const detail = $('model-select-detail');
  const busy = state.modelBusy || state.settingsBusy || state.generating;
  if (action) {
    action.hidden = status === 'loaded';
    action.disabled = !entry || busy || status === 'loaded' || !state.webgpuAvailable;
    action.textContent = status === 'loaded' ? 'Loaded' : modelFit(entry) === 'too-large' ? 'Load anyway' : 'Load';
  }
  if (remove) {
    remove.hidden = status !== 'stored' && status !== 'loaded';
    remove.disabled = busy;
  }
  if (detail) {
    const failed = entry && state.modelLoadNote?.modelId === entry.modelId ? state.modelLoadNote.text : '';
    detail.textContent = !state.webgpuAvailable ? NO_WEBGPU_NOTE : failed || (entry ? buildModelDetail(entry, status) : '');
    detail.hidden = !detail.textContent;
  }
  const retry = $('webgpu-retry');
  if (retry) retry.hidden = state.webgpuAvailable;
  // Without WebGPU no model can load, so the picker would only name a model that is not running.
  const picker = $('model-picker');
  if (picker) picker.classList.toggle('model-picker--unavailable', !state.webgpuAvailable);
  const select = $('model-select');
  if (select) select.disabled = !catalog.length || busy;
  const profile = $('set-profile');
  if (profile) profile.disabled = busy;
  syncSendButton();
}

async function loadSelectedModel({ entry = selectedEntry(), runtimeProfile = state.settings.runtimeProfile, force = false } = {}) {
  if (!entry || state.generating || state.modelBusy || (!force && state.modelId === entry.modelId)) return;
  const previousStatus = state.modelStatus[entry.modelId];
  // Phases after the manifest read or fetch model files; seeing none means no weights moved.
  let sawModelFiles = false;
  state.modelLoadNote = null;
  state.modelBusy = true;
  state.modelStatus[entry.modelId] = 'loading';
  syncModelControls();
  setPhase('Loading model…', { busy: true, quiet: true });
  setProgress({ phase: 'prepare', message: 'Preparing model' });
  try {
    const model = await dr.load(entry.modelId, {
      cache: 'opfs',
      runtimeProfile,
      onProgress: (event) => {
        if (event?.phase && !['resolve', 'manifest'].includes(event.phase)) sawModelFiles = true;
        setProgress(event);
      },
    });
    if (state.model && state.model !== model) {
      await state.model.unload().catch(() => {});
    }
    for (const modelId of Object.keys(state.modelStatus)) {
      if (state.modelStatus[modelId] === 'loaded') state.modelStatus[modelId] = 'stored';
    }
    state.model = model;
    state.bootError = null;
    state.modelId = model.modelId;
    state.modelStatus[entry.modelId] = 'loaded';
    localStorage.setItem(LAST_USED_MODEL_STORAGE_KEY, entry.modelId);
    onModelLoaded?.(model, entry.modelId);
    setPhase('Ready', { quiet: true });
  } catch (error) {
    state.modelStatus[entry.modelId] = previousStatus ?? 'available';
    // Ask what is actually saved before saying anything about downloads.
    const wasStored = previousStatus === 'stored';
    let storedNow = wasStored;
    try { storedNow = (await dr.listPersistentModels()).some((item) => item.modelId === entry.modelId); } catch { /* keep what was known */ }
    if (storedNow) state.modelStatus[entry.modelId] = 'stored';
    state.modelLoadNote = { modelId: entry.modelId, text: loadFailureNote(entry, error, { sawModelFiles, wasStored, storedNow }) };
    if (isNoAdapterError(error)) state.webgpuAvailable = false;
    setPhase('Load failed');
    throw error;
  } finally {
    state.modelBusy = false;
    setProgress(null);
    renderModelOptions();
  }
}

export async function ensurePromptModel() {
  if (state.model) return state.model;
  if (!state.webgpuAvailable) throw new Error('This browser has no WebGPU, so it cannot run a model.');
  const entry = selectedEntry();
  if (!entry) throw new Error('The default model is unavailable. Choose a model above.');
  // Send never starts a download the device is unlikely to be able to use.
  if (modelFit(entry) === 'too-large') {
    throw new Error(`${modelLabel(entry)} needs more memory than this device reports. Choose a smaller model, or press Load anyway.`);
  }
  await loadSelectedModel({ entry });
  if (!state.model) throw new Error('Model loading is already in progress. Try again when ready.');
  return state.model;
}

export async function reloadActiveModel(runtimeProfile) {
  if (!state.model) return;
  const entry = catalog.find((item) => item.modelId === state.modelId);
  await loadSelectedModel({ entry, runtimeProfile, force: true });
}

async function removeSelectedModel() {
  const entry = selectedEntry();
  if (!entry || state.generating || state.modelBusy) return;
  const dialog = $('remove-model-dialog');
  $('remove-model-message').textContent = `Remove ${modelLabel(entry)}?`;
  $('remove-model-detail').textContent = 'Deletes the copy saved in this browser. You can download it again.';
  dialog.returnValue = '';
  const confirmed = new Promise((resolve) => {
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), { once: true });
  });
  dialog.showModal();
  if (!await confirmed) return;
  state.modelBusy = true;
  syncModelControls();
  try {
    if (state.modelId === entry.modelId) {
      await state.model.unload();
      state.modelStatus[entry.modelId] = 'stored';
      state.model = null;
      state.modelId = null;
      state.conversationModelId = null;
      clearConversationHistory();
      setExportEnabled(false);
      setPhase('No model loaded', { quiet: true });
    }
    const removed = await dr.removePersistentModel(entry.modelId);
    if (!removed) throw new Error('The cached model could not be removed. Retry removal.');
    state.modelStatus[entry.modelId] = 'available';
    localStorage.removeItem(LAST_USED_MODEL_STORAGE_KEY);
  } finally {
    state.modelBusy = false;
    renderModelOptions();
  }
}

function bindModelControls() {
  const select = $('model-select');
  if (select && select.dataset.bound !== 'true') {
    select.dataset.bound = 'true';
    select.addEventListener('change', () => {
      selectedModelId = select.value || null;
      state.modelLoadNote = null;
      syncModelControls();
    });
  }
  const action = $('model-select-action');
  if (action && action.dataset.bound !== 'true') {
    action.dataset.bound = 'true';
    action.addEventListener('click', () => {
      loadSelectedModel().catch((error) => setPhase(`Load failed: ${error.message}`));
    });
  }
  const retry = $('webgpu-retry');
  if (retry && retry.dataset.bound !== 'true') {
    retry.dataset.bound = 'true';
    retry.addEventListener('click', () => { void checkWebGpu(); });
  }
  const remove = $('model-select-remove');
  if (remove && remove.dataset.bound !== 'true') {
    remove.dataset.bound = 'true';
    remove.addEventListener('click', () => {
      removeSelectedModel().catch((error) => setPhase(`Remove failed: ${error.message}`));
    });
  }
}

export function canRemoveModelStatus(status) {
  return status === 'stored' || status === 'loaded';
}

// Explain download cost and memory need before loading; the selected model and enabled Send show readiness.
export function buildModelDetail(entry, status = 'available', reportedBytes = reportedMemoryBytes()) {
  if (!entry || status === 'loaded' || status === 'loading') return '';
  const required = entry.quickstartRequirements?.vramRequiredBytes;
  const size = downloadSize(entry);
  if (modelFit(entry, reportedBytes) === 'too-large') {
    return `Needs about ${memorySize(required)} of GPU memory, and this browser reports ${memorySize(reportedBytes)} of system memory, so it will probably fail to load.`
      + (status === 'stored' || !size ? '' : ` Loading it downloads about ${size} first.`);
  }
  const need = Number.isFinite(required) && required > 0 ? ` Needs about ${memorySize(required)} of GPU memory.` : '';
  if (status === 'stored') return `Saved in this browser, so it loads without downloading.${need}`;
  return size ? `First use downloads about ${size}. Your browser keeps a copy if it can.${need}` : need.trim();
}

export function setModelCallbacks(callbacks = {}) {
  onModelLoaded = callbacks.onLoaded ?? null;
  onProgress = callbacks.onDownloadProgress ?? null;
}

export async function loadCatalog() {
  const entries = await dr.listModelDetails();
  catalog = entries.filter(isChatModel).map((entry) => ({ ...entry }));
  selectedModelId = catalog.find((entry) => entry.modelId === DEFAULT_DEMO_MODEL_ID)?.modelId ?? null;
  state.quickModelCatalog = catalog.map((entry) => ({ ...entry }));
  bindModelControls();
  renderModelOptions();
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
  renderModelOptions();
  return registered;
}

export async function loadDefaultStoredModel() {
  const registered = await dr.listPersistentModels();
  const entry = selectDefaultStoredModel(
    catalog.filter((item) => item.modelId === DEFAULT_DEMO_MODEL_ID), registered, DEFAULT_DEMO_MODEL_ID
  );
  if (!entry) return null;
  selectedModelId = entry.modelId;
  renderModelOptions();
  await loadSelectedModel();
  return state.model;
}

export function renderModelOptions() {
  const select = $('model-select');
  if (select) {
    select.replaceChildren();
    for (const entry of catalog) {
      const option = document.createElement('option');
      option.value = entry.modelId;
      option.textContent = [modelLabel(entry), downloadSize(entry),
        modelFit(entry) === 'too-large' ? 'too large for this device' : ''].filter(Boolean).join(' · ');
      select.append(option);
    }
    select.disabled = catalog.length === 0;
    select.value = selectedModelId ?? '';
  }
  const status = $('model-select-status');
  if (status) {
    status.textContent = catalog.length ? '' : 'No supported models';
    status.hidden = catalog.length > 0;
  }
  syncModelControls();
}
