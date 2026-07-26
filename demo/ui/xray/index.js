import { state } from '../state.js';
import { $ } from '../dom.js';

const XRAY_STORAGE_KEY = 'doppler.demo.xray-enabled';
let initialized = false;
let onChangeCallback = null;

function readPreference() {
  try {
    return localStorage.getItem(XRAY_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

function writePreference(value) {
  try {
    localStorage.setItem(XRAY_STORAGE_KEY, String(value));
  } catch {
    // Preferences are optional.
  }
}

function syncState() {
  const enabled = $('xray-toggle-all')?.checked === true;
  state.xrayEnabled = enabled;
  const shell = $('xray-shell');
  const container = $('xray-container');
  if (shell) shell.hidden = !enabled;
  if (container) container.hidden = !enabled;
  writePreference(enabled);
  onChangeCallback?.();
}

export function initXray(options = {}) {
  if (initialized) return;
  initialized = true;
  onChangeCallback = options.onChange ?? null;
  const toggle = $('xray-toggle-all');
  if (toggle) {
    const requested = new URLSearchParams(window.location.search).get('xray');
    toggle.checked = requested === 'all' || (requested == null && readPreference());
    toggle.addEventListener('change', syncState);
  }
  syncState();
}

export function isXrayEnabled() {
  return $('xray-toggle-all')?.checked === true;
}

export function isXrayProfilingNeeded() {
  return isXrayEnabled();
}

export function resetXray() {
  const container = $('xray-container');
  if (container) container.replaceChildren();
}

function addField(container, label, value) {
  const row = document.createElement('div');
  row.className = 'xray-section';
  const heading = document.createElement('div');
  heading.className = 'xray-section-header';
  heading.textContent = label;
  const body = document.createElement('pre');
  body.className = 'xray-content';
  body.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  row.append(heading, body);
  container.append(row);
}

export function updateXrayPanels(receipt = state.lastInspection) {
  if (!isXrayEnabled() || !receipt) return;
  const container = $('xray-container');
  if (!container) return;
  container.replaceChildren();
  addField(container, 'Observation contract', receipt.policy);
  addField(container, 'Execution identity', receipt.fingerprint?.identity?.execution ?? {});
  addField(container, 'Adapter', receipt.fingerprint?.identity?.adapter ?? {});
  addField(container, 'Generation statistics', receipt.generationEvidence?.stats ?? {});
  addField(container, 'Comparison fingerprints', {
    full: receipt.fingerprint?.fullDigest,
    quality: receipt.fingerprint?.qualityDigest,
    performance: receipt.fingerprint?.performanceDigest,
  });
}

export function getXrayRuntimeNoticeText(options = {}) {
  if (options.profilingEnabled) {
    return 'Deep X-Ray modifies execution and enables GPU timestamp queries. Its timings are diagnostic, not representative throughput.';
  }
  if (options.wordQualityEnabled) {
    return 'Guided quality inspection captures token probabilities and changes execution. Compare quality only when the canonical fingerprint matches.';
  }
  return 'Always-on evidence records existing wall timing without GPU timestamp queries. This is the performance-representative observation tier.';
}
