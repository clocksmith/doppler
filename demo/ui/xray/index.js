import { state } from '../state.js';
import { $ } from '../dom.js';
import { renderGpuTimings } from './gpu-timings.js';

const XRAY_STORAGE_KEY = 'doppler.demo.xray-enabled';
let initialized = false;
let onChangeCallback = null;

function readPreference() {
  try {
    const saved = localStorage.getItem(XRAY_STORAGE_KEY);
    return saved === 'true';
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
  if (enabled) updateXrayPanels();
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

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function number(value) {
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function milliseconds(value) {
  return number(value) === null ? 'Not captured' : value.toLocaleString(undefined, {
    maximumFractionDigits: 1,
  }) + ' ms';
}

function fact(value) {
  return typeof value === 'string' && value.trim() ? value : 'Not captured';
}

function panel(title, description) {
  const section = element('section', 'xray-card demo-panel demo-panel--inset');
  section.append(element('h3', '', title));
  if (description) section.append(element('p', 'xray-caption', description));
  return section;
}

function metric(list, label, value) {
  const group = element('div');
  group.append(element('dt', '', label), element('dd', '', value));
  list.append(group);
}

function barRow(label, value, maximum, formatted, kind = '') {
  const row = element('div', 'xray-measure-row');
  const track = element('div', 'xray-measure-track');
  track.setAttribute('aria-hidden', 'true');
  if (number(value) !== null) {
    const fill = element('div', 'xray-measure-fill ' + kind);
    fill.style.width = (maximum > 0 ? Math.min(100, value / maximum * 100) : 0) + '%';
    track.append(fill);
  }
  row.append(element('span', 'xray-measure-label', label), track,
    element('span', 'xray-measure-value', formatted));
  return row;
}

function addTiming(container, receipt) {
  const stats = receipt.generationEvidence?.stats ?? {};
  const total = number(receipt.wallTimingMs);
  const prefill = number(stats.prefillTimeMs);
  const decode = number(stats.decodeTimeMs);
  const section = panel('Where time went',
    'Bars share one scale from zero. They are separate timers, not slices of the total.');
  const metrics = element('dl', 'xray-overview');
  metric(metrics, 'Tokens written', Array.isArray(receipt.generatedTokenIds)
    ? String(receipt.generatedTokenIds.length) : 'Not captured');
  const rate = number(stats.tokensPerSecond) ?? (number(stats.decodeTokens) !== null && decode > 0 ? stats.decodeTokens / (decode / 1000) : null);
  metric(metrics, 'Writing speed', rate === null ? 'Not captured' : rate.toFixed(1) + ' tokens/s');
  section.append(metrics);
  const maximum = Math.max(total ?? 0, prefill ?? 0, decode ?? 0);
  section.append(
    barRow('Total elapsed', total, maximum, milliseconds(total), 'is-total'),
    barRow('Read prompt', prefill, maximum, milliseconds(prefill), 'is-prefill'),
    barRow('Write answer', decode, maximum, milliseconds(decode), 'is-decode')
  );
  const axis = element('div', 'xray-measure-axis');
  axis.append(element('span', '', '0 ms'), element('span', '', milliseconds(maximum)));
  if (total !== null || prefill !== null || decode !== null) section.append(axis);
  if (total !== null && prefill !== null && decode !== null && prefill + decode > total) {
    section.append(element('p', 'xray-diagnostic',
      'The read and write timers add up to more than the total, so they overlap and cannot be summed.'));
  }
  container.append(section);
}

function addIdentity(container, receipt) {
  const identity = receipt.fingerprint?.identity ?? {};
  const execution = identity.execution ?? receipt.generationEvidence?.backendIdentity ?? {};
  const stats = receipt.generationEvidence?.stats ?? {};
  const adapter = identity.adapter ?? {};
  const section = panel('What ran', 'Recorded with this answer; it can differ from the model selected now.');
  const facts = element('dl', 'xray-identity');
  metric(facts, 'Model', fact(identity.artifact?.modelId));
  metric(facts, 'Graphics interface', fact(execution.backend));
  metric(facts, 'GPU', [adapter.vendor, adapter.architecture, adapter.device]
    .filter((value) => typeof value === 'string' && value.trim()).join(' / ') || 'Not captured');
  metric(facts, 'Math precision', fact(execution.activationDtype));
  metric(facts, 'GPU program set', fact(execution.kernelPathId ?? stats.kernelPathId));
  metric(facts, 'Recording mode', fact(receipt.policy?.label ?? receipt.policy?.id));
  section.append(facts);
  const notes = element('ul', 'xray-diagnostics');
  notes.append(element('li', '', 'Step times below come from the GPU’s own clock where available. They show where time went, not how busy the GPU was.'));
  notes.append(element('li', '', receipt.fingerprint?.qualityDigest
    ? 'A fingerprint of this answer is saved, so a matching run can be checked against it.'
    : 'No answer fingerprint is saved, so this run cannot be checked against others.'));
  section.append(notes);
  container.append(section);
}

function addRawReceipt(container, receipt) {
  const details = element('details', 'xray-section');
  details.append(element('summary', 'xray-section-header', 'Full run record (JSON)'));
  const body = element('pre', 'xray-content');
  details.append(body);
  details.addEventListener('toggle', () => {
    if (details.open && !body.textContent) body.textContent = JSON.stringify(receipt, null, 2);
  });
  container.append(details);
}

export function updateXrayPanels(receipt = state.lastInspection) {
  if (!isXrayEnabled()) return;
  const container = $('xray-container');
  if (!container) return;
  container.replaceChildren();
  if (!receipt) {
    container.append(element('p', 'xray-empty', state.lastImportedReport
      ? 'This file has no detailed run record. Its text is shown in the conversation.'
      : 'Send a message, or show the recorded answer, to see where time went.'));
    return;
  }
  const source = state.lastImportedReport ? 'Opened file'
    : state.lastRun?.mode === 'sample-inspection' ? 'Recorded answer' : 'This browser';
  const provenance = element('div', 'xray-provenance');
  provenance.append(element('strong', '', source));
  provenance.append(element('span', '', receipt.policy?.performanceRepresentative === true
    ? 'Normal-speed timing. Compare only with runs that have a matching fingerprint.'
    : 'Slowed by measurement, not a speed benchmark'));
  container.append(provenance);
  const grid = element('div', 'xray-dashboard');
  addTiming(grid, receipt);
  addIdentity(grid, receipt);
  container.append(grid);
  renderGpuTimings(container, receipt);
  addRawReceipt(container, receipt);
}

// Says how the selected observation policy changes the next answer.
export function getXrayRuntimeNoticeText(options = {}) {
  if (options.activationRecording && options.profilingEnabled) {
    return 'The next answer records numerical activity and GPU timing. Recording changes execution, so its times are not a speed benchmark.';
  }
  if (options.activationRecording) {
    return 'The next answer records numerical activity. Recording slows the model, so its times are not a speed benchmark.';
  }
  if (options.profilingEnabled) {
    return 'The next answer measures GPU time for each step. Measuring slows the model, so its times are not a speed benchmark.';
  }
  if (options.wordQualityEnabled || options.alternativesRecorded) {
    return 'Every answer saves the alternatives for each token, which slows the model. Times shown are not a speed benchmark.';
  }
  return 'Times are recorded without slowing the model.';
}
