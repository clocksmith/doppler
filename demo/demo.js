import { boot } from './boot.js';
import { reloadActiveModel, setModelCallbacks } from './models.js';
import { initInput, setRunHandler } from './input.js';
import { initSettings } from './settings.js';
import { initReport } from './report.js';
import { onModelLoaded, runGeneration, stopGeneration, loadSampleInspection } from './core.js';
import { SAMPLE_INSPECTION_RECEIPT } from './data/sample-inspection.js';
import { showTokenInspectorView } from './output.js';
import { state } from './ui/state.js';
import { initPrecisionReplay } from './ui/precision-replay/index.js';
import { initXray, getXrayRuntimeNoticeText, isXrayProfilingNeeded } from './ui/xray/index.js';
import { flushPwaLaunchState, initPwa } from './pwa.js';

function $(id) { return document.getElementById(id); }

function refreshRuntimeNotice() {
  const xrayEnabled = $('xray-toggle-all')?.checked === true;
  const wordQualityEnabled = $('set-word-quality')?.checked === true;
  const tokenInspectorActive = state.tokenInspectorActive;
  const summary = document.querySelector('.chat-controls-summary-state');
  if (summary) {
    summary.textContent = [xrayEnabled && 'X-Ray', wordQualityEnabled && 'Word quality', tokenInspectorActive && 'Tokens']
      .filter(Boolean).join(' · ') || 'Standard';
  }
  const xraySummary = $('xray-summary-state');
  if (xraySummary) {
    xraySummary.textContent = xrayEnabled ? 'Enabled · 5 evidence panels' : 'Disabled';
  }
  if (xrayEnabled) {
    const inspectionWorkspace = $('inspection-workspace');
    if (inspectionWorkspace) inspectionWorkspace.open = true;
  }

  const el = $('runtime-notice');
  if (!el) return;
  const text = getXrayRuntimeNoticeText({
    wordQualityEnabled,
    tokenInspectorActive,
    traceEnabled: $('set-trace')?.checked === true,
    profilingEnabled: isXrayProfilingNeeded(),
  });
  el.textContent = text ?? '';
  el.hidden = !xrayEnabled && !wordQualityEnabled && !tokenInspectorActive;
}

async function init() {
  initPwa();

  // Wire model callbacks
  setModelCallbacks({
    onLoaded: onModelLoaded,
    onDownloadProgress: null,
  });

  // Init UI modules
  await initSettings({ requireDefaultProfile: true, onProfileChange: reloadActiveModel });
  initReport();
  await initInput();
  flushPwaLaunchState();

  // Wire run/stop
  setRunHandler(runGeneration);
  $('stop-btn')?.addEventListener('click', stopGeneration);

  // Wire sample inspection
  $('sample-run-btn')?.addEventListener('click', () => {
    loadSampleInspection(SAMPLE_INSPECTION_RECEIPT);
    refreshRuntimeNotice();
  });

  // Wire token inspector toggle
  const inspectorToggle = $('token-inspector-toggle');
  if (inspectorToggle) {
    state.tokenInspectorActive = true;
    inspectorToggle.classList.add('is-active');
    inspectorToggle.setAttribute('aria-pressed', 'true');
    inspectorToggle.addEventListener('click', () => {
      state.tokenInspectorActive = !state.tokenInspectorActive;
      inspectorToggle.classList.toggle('is-active', state.tokenInspectorActive);
      inspectorToggle.setAttribute('aria-pressed', String(state.tokenInspectorActive));
      showTokenInspectorView(state.tokenInspectorActive);
      refreshRuntimeNotice();
    });
  }

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
