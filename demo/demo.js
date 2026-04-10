import { log } from 'doppler-gpu';
import { boot } from './boot.js';
import { setModelCallbacks } from './models.js';
import { initInput, setRunHandler } from './input.js';
import { initSettings } from './settings.js';
import { initReport } from './report.js';
import { onModelLoaded, runGeneration, stopGeneration } from './core.js';
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
  initSettings();
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

  // Boot sequence
  await boot();
}

init().catch((err) => {
  log.error('Demo', `Init failed: ${err.message}`);
});
