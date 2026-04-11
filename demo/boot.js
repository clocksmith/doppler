import { isWebGPUAvailable, initDevice } from 'doppler-gpu/tooling';
import { state } from './ui/state.js';
import { loadCatalog, checkStoredModels, renderModelCards } from './models.js';

function $(id) { return document.getElementById(id); }

function setBootStatus(text) {
  const el = $('boot-status');
  if (el) el.textContent = text;
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
    if (!isWebGPUAvailable()) {
      throw new Error('WebGPU is not available in this browser. Try Chrome 113+ or Edge 113+.');
    }

    // Step 2: Init device
    setBootStatus('Initializing GPU...');
    await initDevice();

    // Step 3: Load catalog
    setBootStatus('Loading model catalog...');
    await loadCatalog();

    // Step 4: Check OPFS for stored models
    setBootStatus('Checking stored models...');
    await checkStoredModels();

    // Step 5: Render and show
    renderModelCards();
    state.phase = 'ready';
    hideOverlay();
  } catch (err) {
    state.phase = 'error';
    state.bootError = err.message;
    setBootStatus('');
    showBootError(err.message);
  }
}
