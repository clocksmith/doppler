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
  const xrayEnabled = $('xray-toggle-all')?.checked === true;
  const wordQualityEnabled = $('set-word-quality')?.checked === true;
  const summary = document.querySelector('.chat-controls-summary-state');
  if (summary) {
    summary.textContent = `X-Ray ${xrayEnabled ? 'on' : 'off'} · Word quality ${wordQualityEnabled ? 'on' : 'off'}`;
  }
  const xraySummary = $('xray-summary-state');
  if (xraySummary) {
    xraySummary.textContent = xrayEnabled ? 'Enabled · 5 evidence panels' : 'Disabled';
  }

  const el = $('runtime-notice');
  if (!el) return;
  const text = getXrayRuntimeNoticeText({
    wordQualityEnabled,
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

  // Init xray (reads URL ?xray= flags, wires the all-panels checkbox)
  try {
    initXray({ onChange: refreshRuntimeNotice });
  } catch {
    // xray init is optional
  }
  $('set-word-quality')?.addEventListener('change', refreshRuntimeNotice);
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
  console.error(`Demo initialization failed: ${err.message}`);
  showInitError(err?.message || String(err));
});
