// Divergence Faultline: where two runs of the same question start to differ.
// The first observed numerical difference and the first different answer token
// are reported separately. A difference is a place to look, not a cause.

import { compareRuns } from '../shared-run.js';
import { $, h, percent, visible, where } from '../viz/dom.js';
import { openInspector } from '../viz/inspect.js';
import { provenanceMark } from '../viz/provenance.js';
import { differenceStrength } from '../viz/scale.js';
import { comparableCoverage } from './comparison-model.js';
import { track } from '../viz/track.js';

export function renderFaultline({ run: a, other: b, selection }) {
  $('compare-view').hidden = !a || !b;
  if (!a || !b) return;
  const tolerance = Number($('tolerance').value);
  const result = compareRuns(a, b, { tolerance: Number.isFinite(tolerance) && tolerance >= 0 ? tolerance : 1e-3 });
  $('compare-title').textContent = `Compare run ${a.id.slice(4, 10)} with run ${b.id.slice(4, 10)}`;
  $('compare-blockers').replaceChildren(...result.blockers.map((text) => h('li', {}, text)));
  $('circumstances').hidden = result.circumstances.length === 0;
  const text = (value) => (value === null ? 'not recorded' : typeof value === 'object' ? Object.values(value).filter(Boolean).join(' ') : String(value));
  $('circumstances').tBodies[0].replaceChildren(...result.circumstances.map((row) => h('tr', {}, h('td', {}, row.field), h('td', {}, text(row.a)), h('td', {}, text(row.b)))));
  $('compare-layers').hidden = true;
  $('compare-layers-note').textContent = '';
  $('compare-tokens').replaceChildren();
  if (!result.comparable) {
    $('compare-verdict').textContent = 'These two runs are not the same question, so they cannot be compared.';
    $('compare-verdict-row').className = 'status-row warn';
    return;
  }
  const { tokens, layers } = result;
  const same = tokens.firstDifference === null && !layers?.firstAboveTolerance;
  $('compare-verdict').textContent = same ? 'No difference observed.' : 'The runs differ.';
  $('compare-verdict-row').className = same ? 'status-row ready' : 'status-row';
  $('compare-tokens').replaceChildren(provenanceMark('measured', 'The tokens each run produced.'), ' ', tokens.firstDifference === null
    ? `Both runs produced the same ${tokens.lengths[0]} tokens.`
    : `The answers first differ at token ${tokens.firstDifference + 1}: "${visible(tokens.a?.text ?? null)}" (${percent(tokens.a?.probability ?? null)}) in the shown run, "${visible(tokens.b?.text ?? null)}" (${percent(tokens.b?.probability ?? null)}) in the other.`);
  if (tokens.firstDifference !== null && tokens.a) $('compare-tokens').append(' ', h('button', { type: 'button', className: 'btn btn-small', on: { click: () => selection.select({ position: a.inputTokenIds.length + tokens.firstDifference }) } }, 'Select first different token'));
  if (!layers) { $('compare-layers-note').textContent = result.layersUnavailable; return; }
  $('compare-layers').hidden = false;
  const first = layers.firstAboveTolerance;
  const coverage = comparableCoverage(layers);
  const promptLength = a.inputTokenIds.length;
  const place = (position) => (position < promptLength ? `input token ${position + 1}` : `answer token ${position - promptLength + 1}`);
  $('compare-first').replaceChildren(provenanceMark('derived', 'Computed from the layer outputs both runs recorded.'), ' ', first
    ? `First observed difference: layer ${first.layer + 1} at ${place(first.position)}, ${first.relative.toExponential(1)} relative.`
    : layers.firstNonzero
      ? `Nothing above ${layers.tolerance}. The first difference of any size is at layer ${layers.firstNonzero.layer + 1}, ${place(layers.firstNonzero.position)}: ${layers.firstNonzero.relative.toExponential(1)} relative.`
      : 'Every compared layer output is identical.');
  $('compare-layers-note').textContent = `${layers.positions.length} token positions × ${layers.layerCount} layers compared`
    + `${tokens.firstDifference === null ? '' : ', up to the last position where both runs were given the same tokens'}`
    + `${layers.unrecorded ? `; ${layers.unrecorded} cells were not recorded in one of the runs and are hatched` : ''}. "First observed" is first among what both runs recorded.`;

  $('compare-layers-note').append(` Coverage: ${coverage.measured}/${coverage.expected} comparable cells recorded. Missing cells cannot establish agreement.`);
  if (!coverage.measured) $('compare-first').textContent = 'No layer output cells were recorded in common; numerical agreement is unknown.';
  if (first) $('compare-first').append(' ', h('button', { type: 'button', className: 'btn btn-small', on: { click: () => selection.select({ position: first.position, layer: first.layer }) } }, 'Select first numerical difference'));
  const chosen = selection.get();
  // The layer axis: each layer's largest difference over the compared positions.
  const peak = (layer) => {
    const values = layers.relative.map((row) => row[layer]).filter((value) => value !== null);
    return values.length ? Math.max(...values) : null;
  };
  $('fault-track').replaceChildren(track({
    count: layers.layerCount,
    selected: chosen.layer,
    label: 'Layers, shaded by their largest difference',
    strength: (layer) => { const value = peak(layer); return value === null ? null : differenceStrength(value); },
    title: (layer) => { const value = peak(layer); return `Layer ${layer + 1}: ${value === null ? 'not recorded' : value === 0 ? 'identical' : `largest difference ${value.toExponential(1)} relative`}`; },
    onSelect: (layer) => selection.select({ layer }),
  }));

  const inspect = (position, layer, value) => {
    selection.select({ position, layer });
    openInspector({
      title: `Layer ${layer + 1} at ${place(position)}`,
      value: value === 0 ? 'identical' : `${value.toExponential(3)} relative difference`,
      provenance: 'derived',
      rows: [['Rule', 'distance between the two runs\' layer outputs, divided by the larger of their sizes']],
      source: `layer outputs recorded by ${a.observation.policyId ?? 'an unrecorded tier'} and ${b.observation.policyId ?? 'an unrecorded tier'}`,
      precision: [a.environment.activationDtype, b.environment.activationDtype].map((dtype) => dtype ?? 'not recorded').join(' and '),
      device: `${where(a)} and ${where(b)}`,
    }, null);
  };
  const rows = [h('tr', {}, h('th', {}, ''), ...layers.positions.map((position) => h('th', { title: place(position), attrs: position === chosen.position ? { 'aria-current': 'true' } : {} }, position < promptLength ? `i${position + 1}` : `a${position - promptLength + 1}`)))];
  for (let layer = 0; layer < layers.layerCount; layer++) {
    rows.push(h('tr', { attrs: layer === chosen.layer ? { 'aria-current': 'true' } : {} }, h('th', {}, `layer ${layer + 1}`), ...layers.positions.map((position, p) => {
      const value = layers.relative[p][layer];
      if (value === null) return h('td', { className: 'site-not-recorded', title: `${place(position)}, layer ${layer + 1}: not recorded` });
      return h('td', {
        className: first && first.position === position && first.layer === layer ? 'site-first-difference' : '',
        style: `--site-strength: ${differenceStrength(value).toFixed(2)}`,
        title: `${place(position)}, layer ${layer + 1}: ${value === 0 ? 'identical' : `${value.toExponential(1)} relative`}. Select for the exact value.`,
        tabIndex: 0,
        attrs: { role: 'button', 'aria-label': `${place(position)}, layer ${layer + 1}: ${value.toExponential(3)} relative difference`, ...(position === chosen.position && layer === chosen.layer ? { 'aria-pressed': 'true' } : { 'aria-pressed': 'false' }) },
        on: { click: (event) => inspect(position, layer, value), keydown: (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspect(position, layer, value); } } },
      });
    })));
  }
  $('heat').tBodies[0].replaceChildren(...rows);
}

