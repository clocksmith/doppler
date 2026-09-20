import { boot, setBootStatus, stopBootProgress, updateBootModelProgress } from './boot.js';
import { reloadActiveModel, setModelCallbacks } from './models.js';
import { initInput, setRunHandler } from './input.js';
import { initSettings } from './settings.js';
import { initReport } from './report.js';
import { onModelLoaded, runGeneration, stopGeneration, loadSampleInspection } from './core.js';
import { SAMPLE_INSPECTION_RECEIPT } from './data/sample-inspection.js';
import { renderChatMessages, renderWordQuality, showTokenInspectorView, showWordQuality } from './output.js';
import { state } from './ui/state.js';
import { initPrecisionReplay } from './ui/precision-replay/index.js';
import { initXray, getXrayRuntimeNoticeText, isXrayProfilingNeeded } from './ui/xray/index.js';
import { flushPwaLaunchState, initPwa } from './pwa.js';

function $(id) { return document.getElementById(id); }

function refreshRuntimeNotice() {
  const xrayEnabled = $('xray-toggle-all')?.checked === true;
  const wordQualityEnabled = $('set-word-quality')?.checked === true;
  const tokenInspectorActive = state.tokenInspectorActive;

  const xraySummary = $('xray-summary-state');
  if (xraySummary) {
    xraySummary.textContent = xrayEnabled ? 'Timing, tokens, execution' : 'Disabled';
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
    onDownloadProgress: updateBootModelProgress,
  });

  // Init UI modules
  setBootStatus('Loading runtime profile...');
  await initSettings({ requireDefaultProfile: true, onProfileChange: reloadActiveModel });
  setBootStatus('Preparing chat...');
  initReport();
  await initInput();
  flushPwaLaunchState();

  // Wire run/stop
  setRunHandler(runGeneration);
  $('stop-btn')?.addEventListener('click', stopGeneration);

  // Wire sample inspection
  $('chat-thread')?.addEventListener('click', (event) => {
    if (!event.target.closest('#sample-run-btn')) return;
    loadSampleInspection(SAMPLE_INSPECTION_RECEIPT);
    refreshRuntimeNotice();
  });

  // Wire token inspector toggle
  const inspectorToggle = $('token-inspector-toggle');
  if (inspectorToggle) {
    state.tokenInspectorActive = inspectorToggle.checked;
    inspectorToggle.addEventListener('change', () => {
      state.tokenInspectorActive = inspectorToggle.checked;
      showTokenInspectorView(state.tokenInspectorActive);
      refreshRuntimeNotice();
    });
  }

  // Init xray (reads URL ?xray= flags, wires the all-panels checkbox)
  setBootStatus('Preparing inspection tools...');
  try {
    initXray({ onChange: refreshRuntimeNotice });
  } catch {
    // xray init is optional
  }
  $('set-word-quality')?.addEventListener('change', () => {
    const liveMessage = $('live-assistant-message');
    if (liveMessage?.hidden) {
      renderChatMessages(state.conversationHistory);
    } else {
      const receipt = state.lastInspection;
      const quality = state.wordQualityEnabled && receipt?.quality != null;
      if (quality) renderWordQuality(receipt.quality, receipt.outputText);
      showWordQuality(quality);
    }
    refreshRuntimeNotice();
  });
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
  setBootStatus('Initialization failed');
  stopBootProgress(true);
  const errorEl = $('boot-error');
  if (errorEl) {
    errorEl.textContent = message || 'Unable to initialize demo runtime.';
    errorEl.hidden = false;
  }
}

init().catch((err) => {
  console.error(`Demo initialization failed: ${err.message}`);
  showInitError(err?.message || String(err));
});
