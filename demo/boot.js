import { state } from './ui/state.js';
import { setPhase } from './output.js';
import {
  loadCatalog,
  checkStoredModels,
  loadDefaultStoredModel,
  renderModelOptions,
} from './models.js';

function $(id) { return document.getElementById(id); }

let bootStartedAt = null;
let bootTimer = null;
let loadingSavedModel = false;

function refreshBootTiming() {
  const el = $('boot-timing');
  if (!el || bootStartedAt === null) return;
  const now = performance.now();
  const elapsed = Math.floor((now - bootStartedAt) / 1000);
  el.textContent = `${elapsed}s`;
  el.hidden = false;
}

export function setBootStatus(text, detail = '') {
  const now = performance.now();
  if (bootStartedAt === null) bootStartedAt = now;
  const el = $('boot-status');
  if (el && el.textContent !== text) el.textContent = text;
  if (el) el.title = text;
  const detailEl = $('boot-detail');
  if (detailEl) {
    if (detailEl.textContent !== detail) detailEl.textContent = detail;
    detailEl.title = detail;
    detailEl.hidden = !detail;
  }
  $('boot-overlay')?.setAttribute('aria-busy', 'true');
  if (bootTimer === null) bootTimer = setInterval(refreshBootTiming, 3000);
  refreshBootTiming();
}

export function stopBootProgress(failed = false) {
  if (bootTimer !== null) clearInterval(bootTimer);
  bootTimer = null;
  refreshBootTiming();
  const overlay = $('boot-overlay');
  overlay?.setAttribute('aria-busy', 'false');
  if (overlay) overlay.dataset.failed = String(failed);
}

export function updateBootModelProgress(event) {
  if (!loadingSavedModel || !event) return;
  // Public percentages are phase milestones, not measured overall completion.
  // Preserve the loader's actual message rather than displaying a false total.
  const message = typeof event.message === 'string' ? event.message.trim() : '';
  setBootStatus(
    event.phase === 'ready' ? 'Finishing model setup...' : 'Loading saved model...',
    event.phase === 'ready' ? '' : message
  );
}

async function loadSavedModelWithStatus() {
  setBootStatus('Selecting saved model...');
  loadingSavedModel = true;
  try {
    await loadDefaultStoredModel();
  } finally {
    loadingSavedModel = false;
  }
}

function showBootError(message) {
  const el = $('boot-error');
  if (el) {
    el.textContent = message;
    el.hidden = false;
  }
}

function hideOverlay() {
  stopBootProgress();
  const overlay = $('boot-overlay');
  const app = $('app');
  if (overlay) {
    overlay.hidden = true;
  }
  if (app) app.hidden = false;
}

export async function boot() {
  state.phase = 'booting';
  state.bootError = null;

  try {
    // Step 1: WebGPU check. Without it no model can run here, but the recorded
    // example needs none, so the page still opens and says so beside the picker.
    setBootStatus('Checking WebGPU...');
    state.webgpuAvailable = Boolean(globalThis.navigator?.gpu);

    // Step 2: Load catalog
    setBootStatus('Loading model catalog...');
    await loadCatalog();

    // Step 3: Check OPFS for stored models
    setBootStatus('Checking stored models...');
    const storedModels = await checkStoredModels();

    // Step 4: Reuse a saved model before offering remote downloads
    renderModelOptions();
    if (state.webgpuAvailable && storedModels.length > 0) {
      try {
        // dr.load checks manifest/device compatibility before weight materialization.
        await loadSavedModelWithStatus();
      } catch (error) {
        // A saved model is optional startup work. Keep its bytes and the picker usable.
        state.bootError = error?.message || String(error);
      }
    }

    // Step 5: Show the ready chat surface
    state.phase = 'ready';
    setBootStatus('Ready');
    hideOverlay();
    if (state.bootError) {
      const picker = $('model-picker');
      if (picker) picker.open = true;
      setPhase(`Saved model could not load. Choose another model or retry. ${state.bootError}`);
    }
  } catch (err) {
    state.phase = 'error';
    state.bootError = err.message;
    setBootStatus('Startup failed');
    stopBootProgress(true);
    showBootError(err.message);
  }
}
