import { log } from 'doppler-gpu/tooling';
import { boot } from './boot.js';
import { setModelCallbacks } from './models.js';
import { initInput, setRunHandler } from './input.js';
import { initSettings } from './settings.js';
import { initReport } from './report.js';
import { onModelLoaded, runGeneration, stopGeneration } from './core.js';
import { initPrecisionReplay } from './ui/precision-replay/index.js';
import { initXray, getXrayRuntimeNoticeText, isXrayProfilingNeeded } from './ui/xray/index.js';
import { flushPwaLaunchState, initPwa } from './pwa.js';

function $(id) { return document.getElementById(id); }

function refreshRuntimeNotice() {
  const el = $('runtime-notice');
  if (!el) return;
  const text = getXrayRuntimeNoticeText({
    tokenPressEnabled: $('set-token-press')?.checked === true,
    traceEnabled: $('set-trace')?.checked === true,
    profilingEnabled: isXrayProfilingNeeded(),
  });
  el.textContent = text ?? '';
  el.hidden = !text;
}

async function init() {
  initPwa();

  // Wire model callbacks
  setModelCallbacks({
    onLoaded: onModelLoaded,
    onDownloadProgress: null,
  });

  // Init UI modules
  await initSettings({ requireDefaultProfile: true });
  initReport();
  await initInput();
  flushPwaLaunchState();

  // Wire run/stop
  setRunHandler(runGeneration);
  $('stop-btn')?.addEventListener('click', stopGeneration);

  // Init xray (reads URL ?xray= flags, wires per-panel checkboxes)
  try {
    initXray({ onChange: refreshRuntimeNotice });
  } catch {
    // xray init is optional
  }
  $('set-token-press')?.addEventListener('change', refreshRuntimeNotice);
  $('set-trace')?.addEventListener('change', refreshRuntimeNotice);
  refreshRuntimeNotice();

  try {
    await initPrecisionReplay();
  } catch {
    // precision replay is optional
  }

  // Boot sequence
  await boot();
}

function showInitError(message) {
  const statusEl = $('boot-status');
  const errorEl = $('boot-error');
  if (statusEl) {
    statusEl.textContent = 'Initialization failed';
  }
  if (errorEl) {
    errorEl.textContent = message || 'Unable to initialize demo runtime.';
    errorEl.hidden = false;
  }
}

init().catch((err) => {
  log.error('Demo', `Init failed: ${err.message}`);
  showInitError(err?.message || String(err));
});
