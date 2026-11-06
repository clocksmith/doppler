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
    'Reading the prompt is prefill. Writing the answer is decode. Bars share a zero-based scale; timers are not stacked.');
  const metrics = element('dl', 'xray-overview');
  metric(metrics, 'Output tokens', Array.isArray(receipt.generatedTokenIds)
    ? String(receipt.generatedTokenIds.length) : 'Not captured');
  const rate = number(stats.tokensPerSecond) ?? (number(stats.decodeTokens) !== null && decode > 0 ? stats.decodeTokens / (decode / 1000) : null);
  metric(metrics, 'Recorded decode rate', rate === null ? 'Not captured' : rate.toFixed(1) + ' tokens/s');
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
      'Phase timers exceed total elapsed time. They cannot be treated as an additive breakdown.'));
  }
  container.append(section);
}

function addIdentity(container, receipt) {
  const identity = receipt.fingerprint?.identity ?? {};
  const execution = identity.execution ?? receipt.generationEvidence?.backendIdentity ?? {};
  const stats = receipt.generationEvidence?.stats ?? {};
  const adapter = identity.adapter ?? {};
  const section = panel('What ran', 'Identity recorded with this receipt, not the currently selected model.');
  const facts = element('dl', 'xray-identity');
  metric(facts, 'Model', fact(identity.artifact?.modelId));
  metric(facts, 'Backend', fact(execution.backend));
  metric(facts, 'GPU', [adapter.vendor, adapter.architecture, adapter.device]
    .filter((value) => typeof value === 'string' && value.trim()).join(' / ') || 'Not captured');
  metric(facts, 'Activation precision', fact(execution.activationDtype));
  metric(facts, 'Kernel path', fact(execution.kernelPathId ?? stats.kernelPathId));
  metric(facts, 'Observation policy', fact(receipt.policy?.label ?? receipt.policy?.id));
  section.append(facts);
  const notes = element('ul', 'xray-diagnostics');
  notes.append(element('li', '', 'Kernel durations below are GPU timestamp measurements where captured; phase totals do not establish utilization.'));
  notes.append(element('li', '', receipt.fingerprint?.qualityDigest
    ? 'A quality fingerprint is recorded. Comparison still requires a matching receipt.'
    : 'No quality fingerprint is recorded; comparability has not been established.'));
  section.append(notes);
  container.append(section);
}

function addRawReceipt(container, receipt) {
  const details = element('details', 'xray-section');
  details.append(element('summary', 'xray-section-header', 'Raw inspection receipt (JSON)'));
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
      ? 'This imported report has no inspection receipt. Its summary remains in the conversation.'
      : 'Run a prompt or explore the recorded sample to see its X-Ray.'));
    return;
  }
  const source = state.lastImportedReport ? 'Imported receipt'
    : state.lastRun?.mode === 'sample-inspection' ? 'Recorded sample' : 'Current browser run';
  const provenance = element('div', 'xray-provenance');
  provenance.append(element('strong', '', source));
  provenance.append(element('span', '', receipt.policy?.performanceRepresentative === true
    ? 'Representative timing policy; comparisons still require matching fingerprints.'
    : 'Diagnostic timing / not benchmark evidence'));
  container.append(provenance);
  const grid = element('div', 'xray-dashboard');
  addTiming(grid, receipt);
  addIdentity(grid, receipt);
  container.append(grid);
  renderGpuTimings(container, receipt);
  addRawReceipt(container, receipt);
}

export function getXrayRuntimeNoticeText(options = {}) {
  if (options.profilingEnabled) {
    return 'X-Ray captures GPU timestamps and changes execution. Timings are diagnostic, not a throughput benchmark.';
  }
  if (options.wordQualityEnabled || options.alternativesRecorded) {
    return 'Recording token alternatives changes execution. Timings are diagnostic.';
  }
  return 'Always-on evidence records existing wall timing without GPU timestamp queries. This is the performance-representative observation tier.';
}
