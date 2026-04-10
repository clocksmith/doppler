import { log } from 'doppler-gpu';
import { boot } from './boot.js';
import { setModelCallbacks } from './models.js';
import { initInput, setRunHandler } from './input.js';
import { initSettings } from './settings.js';
import { initReport } from './report.js';
import { onModelLoaded, runGeneration, stopGeneration } from './core.js';
import { initXray } from './ui/xray/index.js';
import { flushPwaLaunchState, initPwa } from './pwa.js';

function $(id) { return document.getElementById(id); }

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
    initXray({ onChange: () => {} });
  } catch {
    // xray init is optional
  }

  // Boot sequence
  await boot();
}

init().catch((err) => {
  log.error('Demo', `Init failed: ${err.message}`);
});
