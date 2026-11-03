import { installDebugGlobal } from 'doppler-gpu/tooling/runtime';
import { boot, setBootStatus, stopBootProgress, updateBootModelProgress } from './boot.js';
import { reloadActiveModel, setModelCallbacks } from './models.js';
import { initInput, setRunHandler } from './input.js';
import { initSettings } from './settings.js';
import { initReport } from './report.js';
import { onModelLoaded, runGeneration, stopGeneration, loadSampleInspection, loadIllustrativeExample, syncBranchView } from './core.js';
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
  $('output-area')?.addEventListener('click', (event) => {
    if (!event.target.closest('#sample-run-btn, #inspect-recorded-run')) return;
    loadSampleInspection(SAMPLE_INSPECTION_RECEIPT);
    refreshRuntimeNotice();
  });
  // Wire Branch: the toggle records alternatives; threshold and "all tokens"
  // only change which recorded tokens are offered as branch points.
  const inspectorToggle = $('token-inspector-toggle');
  const branchOptions = $('branch-options');
  const branchThreshold = $('branch-threshold');
  const branchAllTokens = $('branch-all-tokens');
  const syncBranchControls = () => {
    if (branchOptions) branchOptions.hidden = !state.tokenInspectorActive;
    if (branchThreshold) branchThreshold.disabled = state.branchShowAll;
  };
  if (inspectorToggle) {
    state.tokenInspectorActive = inspectorToggle.checked;
    inspectorToggle.addEventListener('change', () => {
      state.tokenInspectorActive = inspectorToggle.checked;
      syncBranchControls();
      showTokenInspectorView(state.tokenInspectorActive);
      refreshRuntimeNotice();
    });
  }
  branchThreshold?.addEventListener('input', () => {
    const percent = Number(branchThreshold.value);
    if (!branchThreshold.checkValidity() || !Number.isFinite(percent)) return;
    state.branchThreshold = percent / 100;
    syncBranchView();
  });
  branchAllTokens?.addEventListener('change', () => {
    state.branchShowAll = branchAllTokens.checked;
    syncBranchControls();
    syncBranchView();
  });
  syncBranchControls();

  // Init xray (reads URL ?xray= flags, wires the all-panels checkbox)
  setBootStatus('Preparing inspection tools...');
  try {
    initXray({ onChange: refreshRuntimeNotice });
  } catch {
    // xray init is optional
  }
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
    loadIllustrativeExample();
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
