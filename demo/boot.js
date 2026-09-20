import { state } from './ui/state.js';
import {
  loadCatalog,
  checkStoredModels,
  loadDefaultStoredModel,
  renderModelCards,
} from './models.js';

function $(id) { return document.getElementById(id); }

let bootStartedAt = null;
let lastBootUpdateAt = null;
let bootTimer = null;
let loadingSavedModel = false;

function refreshBootTiming() {
  const el = $('boot-timing');
  if (!el || bootStartedAt === null) return;
  const now = performance.now();
  const elapsed = Math.floor((now - bootStartedAt) / 1000);
  const sinceUpdate = Math.floor((now - lastBootUpdateAt) / 1000);
  el.textContent = sinceUpdate >= 6
    ? `${elapsed}s elapsed / last loader update ${sinceUpdate}s ago`
    : `${elapsed}s elapsed`;
  el.hidden = false;
}

export function setBootStatus(text, detail = '') {
  const now = performance.now();
  if (bootStartedAt === null) bootStartedAt = now;
  lastBootUpdateAt = now;
  const el = $('boot-status');
  if (el && el.textContent !== text) el.textContent = text;
  const detailEl = $('boot-detail');
  if (detailEl) {
    if (detailEl.textContent !== detail) detailEl.textContent = detail;
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
    overlay.classList.add('fade-out');
    setTimeout(() => { overlay.hidden = true; }, 350);
  }
  if (app) app.hidden = false;
}

export async function boot() {
  state.phase = 'booting';

  try {
    // Step 1: WebGPU check
    setBootStatus('Checking WebGPU...');
    if (!globalThis.navigator?.gpu) {
      throw new Error('WebGPU is not available in this browser. Try Chrome 113+ or Edge 113+.');
    }

    // Step 2: Load catalog
    setBootStatus('Loading model catalog...');
    await loadCatalog();

    // Step 3: Check OPFS for stored models
    setBootStatus('Checking stored models...');
    const storedModels = await checkStoredModels();

    // Step 4: Reuse a saved model before offering remote downloads
    renderModelCards();
    if (storedModels.length > 0) await loadSavedModelWithStatus();

    // Step 5: Show the ready chat surface
    state.phase = 'ready';
    setBootStatus(state.model ? 'Ready' : 'Choose a model');
    hideOverlay();
  } catch (err) {
    state.phase = 'error';
    state.bootError = err.message;
    setBootStatus('Startup failed');
    stopBootProgress(true);
    showBootError(err.message);
  }
}
