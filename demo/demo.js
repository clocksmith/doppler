import { installDebugGlobal } from 'doppler-gpu/tooling/runtime';
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

// Paint travels independently of button geometry, and completes after release.
function initButtonSplash() {
  const splashes = new WeakMap();
  const splash = (event) => {
    if (event.type === 'click' && event.detail !== 0) return;
    if (event.type === 'pointerdown' && event.button !== 0) return;
    const control = event.target.closest('#app .btn, #app .precision-replay-mode-btn, .confirm-dialog .btn');
    if (!control || control.disabled || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    splashes.get(control)?.cancel();
    splashes.set(control, control.animate([
      { clipPath: 'inset(0 100% 0 0)', opacity: .65, offset: 0 },
      { clipPath: 'inset(0 0 0 0)', opacity: .65, offset: .7 },
      { clipPath: 'inset(0 0 0 0)', opacity: 0, offset: 1 },
    ], { pseudoElement: '::after', duration: 460, easing: 'cubic-bezier(.2,.7,.2,1)' }));
  };
  document.addEventListener('pointerdown', splash, true);
  document.addEventListener('click', splash, true);
}

async function init() {
  installDebugGlobal();
  initButtonSplash();
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
  if (state.phase === 'ready' && !state.model && !state.conversationHistory.length) {
    loadSampleInspection(SAMPLE_INSPECTION_RECEIPT);
    refreshRuntimeNotice();
  }
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
