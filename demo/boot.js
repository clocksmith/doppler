import { state } from './ui/state.js';
import {
  loadCatalog,
  checkStoredModels,
  loadDefaultStoredModel,
  renderModelCards,
} from './models.js';

function $(id) { return document.getElementById(id); }

function setBootStatus(text) {
  const el = $('boot-status');
  if (el) el.textContent = text;
}

async function loadSavedModelWithStatus() {
  const startedAt = performance.now();
  setBootStatus('Loading saved model...');
  const timer = setInterval(() => {
    const progress = state.downloadProgress;
    const details = [];
    if (typeof progress?.message === 'string' && progress.message.trim()) {
      details.push(progress.message.trim());
    }
    if (typeof progress?.percent === 'number' && Number.isFinite(progress.percent)) {
      details.push(`${Math.round(Math.max(0, Math.min(100, progress.percent)))}%`);
    }
    details.push(`${Math.floor((performance.now() - startedAt) / 1000)}s elapsed`);
    setBootStatus(`Loading saved model... ${details.join(' - ')}`);
  }, 3000);
  try {
    await loadDefaultStoredModel();
  } finally {
    clearInterval(timer);
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
    await checkStoredModels();

    // Step 4: Reuse a saved model before offering remote downloads
    renderModelCards();
    await loadSavedModelWithStatus();

    // Step 5: Show the ready chat surface
    state.phase = 'ready';
    hideOverlay();
  } catch (err) {
    state.phase = 'error';
    state.bootError = err.message;
    setBootStatus('');
    showBootError(err.message);
  }
}
