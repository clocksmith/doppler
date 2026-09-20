import { state } from '../state.js';
import { $ } from '../dom.js';

const XRAY_STORAGE_KEY = 'doppler.demo.xray-enabled';
let initialized = false;
let onChangeCallback = null;

function readPreference() {
  try {
    const saved = localStorage.getItem(XRAY_STORAGE_KEY);
    return saved === null ? true : saved === 'true';
  } catch {
    return true;
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

function probability(value) {
  return number(value) !== null && value <= 1 ? value : null;
}

function milliseconds(value) {
  return number(value) === null ? 'Not captured' : value.toLocaleString(undefined, {
    maximumFractionDigits: 1,
  }) + ' ms';
}

function percent(value) {
  return probability(value) === null ? 'Not captured' : (value * 100).toFixed(1) + '%';
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
  metric(metrics, 'Total elapsed', milliseconds(total));
  metric(metrics, 'Reported decode rate', number(stats.tokensPerSecond) === null
    ? 'Not captured' : stats.tokensPerSecond.toFixed(1) + ' tokens/s');
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
  } else if (prefill !== null && decode !== null && prefill !== decode) {
    section.append(element('p', 'xray-diagnostic', decode > prefill
      ? 'Writing the answer took longer than reading the prompt.'
      : 'Reading the prompt took longer than writing the answer.'));
  }
  container.append(section);
}

function svgElement(tag, attributes, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text != null) node.textContent = text;
  return node;
}

function addTokenChart(container, receipt) {
  const tokens = Array.isArray(receipt.tokens) ? receipt.tokens : [];
  const section = panel('How likely was each token?',
    'The model probability of the token it produced, before sampling filters. This measures surprise, not correctness.');
  if (!tokens.length) {
    section.append(element('p', 'xray-empty', 'Token probabilities were not captured for this run.'));
    container.append(section);
    return;
  }
  const chart = svgElement('svg', {
    viewBox: '0 0 640 180', class: 'xray-probability-chart',
    role: 'img', 'aria-label': 'Selected-token probability from 0 to 100 percent, by token position',
    preserveAspectRatio: 'none',
  });
  for (const [value, y] of [[100, 14], [50, 78], [0, 142]]) {
    chart.append(svgElement('line', { x1: 52, x2: 620, y1: y, y2: y, class: 'xray-chart-grid' }));
    chart.append(svgElement('text', { x: 42, y: y + 4, 'text-anchor': 'end' }, value + '%'));
  }
  chart.append(svgElement('text', { x: 52, y: 170 }, 'Token 1'));
  chart.append(svgElement('text', { x: 620, y: 170, 'text-anchor': 'end' }, 'Token ' + tokens.length));
  const xAt = (index) => tokens.length === 1 ? 336 : 52 + index / (tokens.length - 1) * 568;
  let path = '';
  let connected = false;
  let captured = 0;
  for (const [index, token] of tokens.entries()) {
    const value = probability(token?.probability);
    if (value === null) {
      connected = false;
      continue;
    }
    captured++;
    path += (connected ? ' L ' : ' M ') + xAt(index) + ' ' + (142 - value * 128);
    connected = true;
  }
  chart.append(svgElement('path', { d: path, class: 'xray-chart-line' }));
  const cursor = svgElement('line', { x1: 52, x2: 52, y1: 14, y2: 142, class: 'xray-chart-cursor' });
  const point = svgElement('circle', { cx: 52, cy: 142, r: 4, class: 'xray-chart-point' });
  chart.append(cursor, point);
  const coverage = element('p', 'xray-caption',
    captured + ' of ' + tokens.length + ' token probabilities captured. Gaps are missing measurements.');
  const label = element('label', 'xray-token-label', 'Inspect token');
  label.htmlFor = 'xray-token-position';
  const slider = element('input', 'xray-token-slider');
  slider.id = 'xray-token-position';
  slider.type = 'range';
  slider.min = '0';
  slider.max = String(tokens.length - 1);
  slider.step = '1';
  slider.value = '0';
  slider.disabled = tokens.length === 1;
  const readout = element('p', 'xray-token-readout');
  readout.setAttribute('aria-live', 'polite');
  const candidates = element('div', 'xray-candidates');
  const renderSelection = () => {
    const index = Number(slider.value);
    const token = tokens[index];
    const value = probability(token?.probability);
    const tokenText = typeof token?.text === 'string' ? JSON.stringify(token.text) : 'Unrecorded token';
    const description = 'Token ' + (index + 1) + ': ' + tokenText + ' / ' + percent(value);
    readout.textContent = description;
    slider.setAttribute('aria-valuetext', description);
    cursor.setAttribute('x1', xAt(index));
    cursor.setAttribute('x2', xAt(index));
    point.setAttribute('cx', xAt(index));
    point.setAttribute('cy', value === null ? 142 : 142 - value * 128);
    point.style.display = value === null ? 'none' : '';
    candidates.replaceChildren(element('h4', '', 'Other candidates at this position'));
    const choices = Array.isArray(token?.topCandidates) ? token.topCandidates : [];
    if (!choices.length) {
      candidates.append(element('p', 'xray-empty', 'Candidate scores were not captured.'));
      return;
    }
    for (const candidate of choices) {
      const chosen = Number.isInteger(token?.tokenId) && candidate?.tokenId === token.tokenId;
      const name = (typeof candidate?.text === 'string' ? JSON.stringify(candidate.text) : 'Unrecorded token')
        + (chosen ? ' (selected)' : '');
      const p = probability(candidate?.probability);
      candidates.append(barRow(name, p, 1, percent(p), chosen ? 'is-selected' : ''));
    }
    candidates.append(element('p', 'xray-caption',
      'Each bar uses the full 0-100% scale. Only captured candidates are shown; they need not sum to 100%.'));
  };
  slider.addEventListener('input', renderSelection);
  chart.addEventListener('click', (event) => {
    const bounds = chart.getBoundingClientRect();
    if (bounds.width <= 0) return;
    const x = (event.clientX - bounds.left) / bounds.width * 640;
    slider.value = String(Math.round(Math.max(0, Math.min(1, (x - 52) / 568)) * (tokens.length - 1)));
    renderSelection();
  });
  renderSelection();
  section.append(chart, coverage, label, slider, readout, candidates);
  container.append(section);
}

function addIdentity(container, receipt) {
  const identity = receipt.fingerprint?.identity ?? {};
  const execution = identity.execution ?? {};
  const adapter = identity.adapter ?? {};
  const section = panel('What ran', 'Identity recorded with this receipt, not the currently selected model.');
  const facts = element('dl', 'xray-identity');
  metric(facts, 'Model', fact(identity.artifact?.modelId));
  metric(facts, 'Backend', fact(execution.backend));
  metric(facts, 'GPU', [adapter.vendor, adapter.architecture, adapter.device]
    .filter((value) => typeof value === 'string' && value.trim()).join(' / ') || 'Not captured');
  metric(facts, 'Activation precision', fact(execution.activationDtype));
  metric(facts, 'Kernel path', fact(execution.kernelPathId));
  metric(facts, 'Observation policy', fact(receipt.policy?.label ?? receipt.policy?.id));
  section.append(facts);
  const notes = element('ul', 'xray-diagnostics');
  if (receipt.policy?.modifiesExecution === true) {
    notes.append(element('li', '', 'Inspection changed execution. Do not treat these timings as normal chat speed.'));
  }
  notes.append(element('li', '', 'Kernel-level bottlenecks and GPU utilization cannot be inferred from phase totals.'));
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
  addTokenChart(container, receipt);
  addRawReceipt(container, receipt);
}

export function getXrayRuntimeNoticeText(options = {}) {
  if (options.profilingEnabled) {
    return 'X-Ray captures GPU timestamps and changes execution. Timings are diagnostic, not a throughput benchmark.';
  }
  if (options.wordQualityEnabled || options.tokenInspectorActive) {
    return 'Token inspection changes execution. Compare quality only with matching comparison fingerprints.';
  }
  return 'Always-on evidence records existing wall timing without GPU timestamp queries. This is the performance-representative observation tier.';
}
