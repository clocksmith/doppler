import { installDebugGlobal } from 'doppler-gpu/tooling/runtime';
import { boot, setBootStatus, stopBootProgress, updateBootModelProgress } from './boot.js';
import { reloadActiveModel, setModelCallbacks } from './models.js';
import { getIllustrativeExample, initInput, preloadPrompt, setRunHandler } from './input.js';
import { initSettings } from './settings.js';
import { initReport } from './report.js';
import { onModelLoaded, runGeneration, stopGeneration, showRecordedActivity, updateRecordedActivity } from './core.js';
import { state } from './ui/state.js';
import { initPrecisionReplay } from './ui/precision-replay/index.js';
import { initLivePrecision } from './ui/precision-replay/live.js';
import { initActivationGrid, isActivationRecordingEnabled, loadActivityExample, cancelActivityExample } from './ui/activation-grid/index.js';
import { initXray, getXrayRuntimeNoticeText, isXrayProfilingNeeded } from './ui/xray/index.js';
import { flushPwaLaunchState, initPwa } from './pwa.js';

function $(id) { return document.getElementById(id); }

function refreshRuntimeNotice() {
  const xrayEnabled = $('xray-toggle-all')?.checked === true;
  const wordQualityEnabled = $('set-word-quality')?.checked === true;

  // Restoring a measurement preference must not expand the page on every update.
  const measurement = $('activity-measurement-note');
  if (measurement) measurement.textContent = xrayEnabled
    ? 'GPU timing on · adds overhead.' : isActivationRecordingEnabled() ? 'Adds measurement overhead.' : 'Recording off.';

  const el = $('runtime-notice');
  if (!el) return;
  const text = getXrayRuntimeNoticeText({
    activationRecording: isActivationRecordingEnabled(),
    wordQualityEnabled,
    // Every live answer records alternatives, so execution is always observed.
    alternativesRecorded: true,
    traceEnabled: $('set-trace')?.checked === true,
    profilingEnabled: isXrayProfilingNeeded(),
  });
  el.textContent = text ?? '';
  el.hidden = !text;
}

// Each checkbox has an explicit observation policy, including the combined mode.
function initRecordingChoice() {
  if (new URLSearchParams(location.search).get('xray') === 'all') $('run-details').open = true;
  for (const id of ['activation-record', 'xray-toggle-all']) {
    $(id)?.addEventListener('change', refreshRuntimeNotice);
  }
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
  initActivationGrid({ onOpenRecording: showRecordedActivity, onPlayback: updateRecordedActivity });
  initLivePrecision();
  await initInput();
  flushPwaLaunchState();

  // Wire run/stop
  setRunHandler(runGeneration);
  $('stop-btn')?.addEventListener('click', stopGeneration);

  // User actions take ownership before an asynchronous example can replace them.
  $('prompt-input')?.addEventListener('input', cancelActivityExample);
  $('model-select')?.addEventListener('change', cancelActivityExample);
  for (const id of ['run-btn', 'shuffle-btn', 'model-select-action', 'clear-history-btn', 'stop-btn']) {
    $(id)?.addEventListener('click', cancelActivityExample, true);
  }
  $('prompt-input')?.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) cancelActivityExample();
  }, true);
  // Init xray (reads URL ?xray= flags, wires the all-panels checkbox)
  setBootStatus('Preparing inspection tools...');
  try {
    initXray({ onChange: refreshRuntimeNotice });
  } catch {
    // xray init is optional
  }
  $('set-trace')?.addEventListener('change', refreshRuntimeNotice);
  initRecordingChoice();
  refreshRuntimeNotice();

  try {
    await initPrecisionReplay();
  } catch {
    // precision replay is optional
  }

  // Boot sequence
  await boot();
  // A cached model must not suppress the ready-to-run prompt on a return visit.
  const example = getIllustrativeExample();
  if (state.phase === 'ready') {
    if (example) preloadPrompt(example.prompt);
    void loadActivityExample(undefined, { autoplay: true });
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
