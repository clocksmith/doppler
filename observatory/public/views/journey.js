// Layer Journey: how the selected token's representation moved through the
// layers, and what each layer leaned toward. The first is measured activity:
// how far each layer's output is from the one before. It is not a cause. The
// second is a projection the runtime computes on request, where it can; it is
// labelled projected and never shown as words the model produced at a layer.

import { answerIndex, choiceVectors, layerChanges, layerVector } from '../shared-run.js';
import { $, h, percent, visible, where } from '../viz/dom.js';
import { gap } from '../viz/gap.js';
import { openInspector } from '../viz/inspect.js';
import { provenanceMark } from '../viz/provenance.js';
import { magnitudeShare } from '../viz/scale.js';
import { track } from '../viz/track.js';
import { coordinateJourney, rotatePoint } from './spatial-data.js';

// What each layer leaned toward when the model chose the selected token: every
// layer's output at the choosing position, passed through the model's own final
// normalization and output weights by the runtime. A projection, never a
// generated word. `projection` is { state, rows, reason } from the page.
function leaning(run, index, projection, actions) {
  if (index === null) return h('p', { className: 'demo-note' }, 'Projections are shown for answer tokens.');
  const token = run.tokens[index];
  if (projection?.rows) {
    const last = projection.rows.at(-1).candidates[0];
    return [
      h('p', {}, provenanceMark('projected', 'Each layer\'s output passed through the final normalization and output weights.'),
        ` What each layer leaned toward at "${visible(token.text)}". Projections: no layer generated these words.`),
      h('div', { className: 'site-scroll' }, h('table', { id: 'journey-projection' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Layer'), h('th', {}, 'Leaned toward'), h('th', { className: 'site-num' }, 'Probability'), h('th', {}, 'Next two'))),
        h('tbody', {}, ...projection.rows.map((row, layer) => h('tr', {},
          h('td', {}, String(layer + 1)),
          h('td', {}, h('code', {}, visible(row.candidates[0].text)), row.candidates[0].tokenId === token.tokenId ? ' = the token chosen' : ''),
          h('td', { className: 'site-num' }, percent(row.candidates[0].probability)),
          h('td', { className: 'demo-note' }, row.candidates.slice(1, 3).map((candidate) => `${visible(candidate.text)} ${percent(candidate.probability)}`).join(' · ')),
        ))))),
      h('p', { className: 'demo-note' }, last.tokenId === token.tokenId
        ? `Check: the last layer's projection is the token the model chose (${percent(last.probability)} here; the run recorded ${percent(token.probability)}).`
        : `The last layer's projection ("${visible(last.text)}") is not the token the run recorded. Sampling, or a different runtime than the one that recorded the run, can cause that.`),
    ];
  }
  if (!choiceVectors(run, index)) return gap({ what: 'This run did not record every layer at the position where this token was chosen, so there is nothing complete to project.' });
  return gap({
    kind: 'unrun',
    what: projection?.state === 'running' ? 'Projecting…' : (projection?.reason ?? 'Which words each layer leaned toward has not been computed for this token.'),
    action: { label: 'Project the layers', unavailable: actions.projectUnavailable(run), run: () => actions.project(run, index) },
  });
}

// Camera choice is presentation state only. Shared run/token/layer selection
// continues to belong to the page selection, including clicks in this plot.
let cameraAngle = 0.5;
let spatial = true;
function spatialJourney(run, position, selected, inspect) {
  const vectors = Array.from({ length: run.layerOutputs.layers }, (_, layer) => layerVector(run, position, layer));
  const projection = coordinateJourney(vectors);
  if (!projection) return gap({ what: 'No finite, consistently sized layer vectors are available for this position.' });
  const drawing = h('div');
  const svg = (tag, attrs = {}, text = null) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== null) node.textContent = text;
    return node;
  };
  const draw = () => {
    // The path is fitted to the drawing: one scale for both axes, so distances
    // keep their proportions, centred on the points and with the height taken
    // from the path's own shape. A path that barely moves on one axis gets a
    // shallow drawing, not a thin strip in a tall empty box.
    const rotated = projection.points.map((point) => (point ? rotatePoint(point.values, cameraAngle, spatial) : null));
    const present = rotated.filter(Boolean);
    const xs = present.map((point) => point[0]), ys = present.map((point) => point[1]);
    const minX = Math.min(...xs), minY = Math.min(...ys);
    const spanX = Math.max(...xs) - minX || 1, spanY = Math.max(...ys) - minY || 1;
    const width = 640, pad = 40, tallest = 340, shallowest = 170;
    const scale = Math.min((width - 2 * pad) / spanX, (tallest - 2 * pad) / spanY);
    const height = Math.min(tallest, Math.max(shallowest, spanY * scale + 2 * pad));
    const left = (width - spanX * scale) / 2, bottom = height - (height - spanY * scale) / 2;
    const coordinates = rotated.map((point) => (point ? [left + (point[0] - minX) * scale, bottom - (point[1] - minY) * scale] : null));
    const chart = svg('svg', { viewBox: `0 0 ${width} ${Math.round(height)}`, role: 'group', 'aria-label': 'Coordinate projection of recorded layer outputs', style: 'display:block;width:100%;max-width:760px;color:inherit' });
    // The two axes cross at zero, held at the edge when zero lies outside the drawing.
    const zeroX = Math.min(width - 12, Math.max(12, left - minX * scale)), zeroY = Math.min(height - 12, Math.max(12, bottom + minY * scale));
    chart.append(svg('line', { x1: 12, y1: zeroY, x2: width - 12, y2: zeroY, stroke: 'currentColor', opacity: 0.15 }), svg('line', { x1: zeroX, y1: 12, x2: zeroX, y2: height - 12, stroke: 'currentColor', opacity: 0.15 }));
    // Layer numbers are drawn where they do not sit on top of one already drawn;
    // the first, last and selected layers are always numbered.
    const labelled = [];
    const mayLabel = (layer, point) => {
      const always = layer === selected || layer === 0 || layer === coordinates.length - 1;
      if (!always && labelled.some((other) => Math.abs(other[0] - point[0]) < 18 && Math.abs(other[1] - point[1]) < 13)) return false;
      labelled.push(point);
      return true;
    };
    for (let layer = 0; layer < coordinates.length; layer++) {
      const point = coordinates[layer];
      if (!point) continue;
      const before = coordinates[layer - 1];
      if (before) chart.append(svg('line', { x1: before[0], y1: before[1], x2: point[0], y2: point[1], stroke: 'currentColor', opacity: 0.4, 'stroke-width': 2 }));
      const mark = svg('circle', { cx: point[0], cy: point[1], r: layer === selected ? 8 : 5, fill: 'currentColor', opacity: layer === selected ? 1 : 0.65, role: 'button', tabindex: 0, 'aria-label': `Inspect layer ${layer + 1}`, 'aria-pressed': String(layer === selected), style: 'cursor:pointer' });
      mark.append(svg('title', {}, `Layer ${layer + 1}: ${projection.axes.map((axis, i) => `channel ${axis}: ${vectors[layer][axis].toPrecision(5)}`).join(', ')}`));
      mark.addEventListener('click', () => inspect(layer));
      mark.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspect(layer); } });
      chart.append(mark);
      if (mayLabel(layer, point)) chart.append(svg('text', { x: point[0] + 8, y: point[1] - 8, fill: 'currentColor', 'font-size': 11, 'pointer-events': 'none' }, String(layer + 1)));
    }
    drawing.replaceChildren(chart);
  };
  const mode = h('select', { ariaLabel: 'Layer journey projection dimensions', on: { change: (event) => { spatial = event.target.value === '3'; rotation.disabled = !spatial; draw(); } } }, h('option', { value: '3', selected: spatial }, '3D coordinate view'), h('option', { value: '2', selected: !spatial }, '2D coordinate view'));
  const rotation = h('input', { type: 'range', min: -180, max: 180, step: 1, value: Math.round(cameraAngle * 180 / Math.PI), disabled: !spatial, ariaLabel: 'Rotate layer journey', on: { input: (event) => { cameraAngle = Number(event.target.value) * Math.PI / 180; draw(); } } });
  draw();
  return h('section', { className: 'site-spatial-journey' },
    h('p', {}, provenanceMark('derived'), ' Spatial layer journey'),
    h('p', { className: 'demo-note' }, `Channels ${projection.axes.join(', ')}${projection.varianceShare === null ? ' (no variation here)' : ` · ${(projection.varianceShare * 100).toFixed(1)}% of the variance at this token`}. A coordinate view, not a semantic map or a cause; axes differ per token.`),
    h('label', {}, 'View ', mode), h('label', {}, ' Rotate ', rotation), drawing,
    h('p', { className: 'demo-note' }, 'Numbers are layers; crowded ones are left unnumbered. Select a point, or use the layer track below.'),
  );
}

export function renderJourney({ run, selection, projections = null, actions = null }) {
  const { position, layer: chosen } = selection.get();
  const root = $('journey');
  if (position === null) { root.replaceChildren(h('p', { className: 'demo-note' }, 'Select a token in the answer to follow it through the layers.')); return; }
  const index = answerIndex(run, position);
  const name = index === null ? `input token ${position + 1}` : `answer token ${index + 1}, "${visible(run.tokens[index].text)}"`;
  const changes = layerChanges(run, position);
  if (!changes) {
    root.replaceChildren(gap({ what: run.layerOutputs ? `This run did not record layer outputs at ${name}.` : 'This run was recorded without layer outputs.' }));
    return;
  }
  const largest = Math.max(0, ...changes.map((row) => row.change ?? 0));
  const inspect = (layer) => {
    selection.select({ layer });
    if (layer === null) return;
    const row = changes[layer];
    openInspector({
      title: `Layer ${layer + 1} at ${name}`,
      value: row.change === null ? null : `${row.change.toExponential(3)} relative change from layer ${layer}`,
      provenance: row.change === null ? 'missing' : 'derived',
      rows: [['Size of this layer\'s output', row.size === null ? null : row.size.toPrecision(5)], ['Rule', 'distance from the previous layer\'s output, divided by this output\'s size']],
      source: `layer outputs recorded by ${run.observation.policyId ?? 'an unrecorded tier'}`,
      precision: run.environment.activationDtype,
      device: where(run),
    });
  };
  root.replaceChildren(
    spatialJourney(run, position, chosen, inspect),
    h('p', {}, provenanceMark('derived', 'Computed from the layer outputs this run recorded.'), ` How far each layer moved the representation at ${name}. Darker is a larger change. This is activity, not cause.`),
    track({
      count: changes.length,
      selected: chosen,
      label: `Change per layer at ${name}`,
      strength: (layer) => (changes[layer].change === null ? null : magnitudeShare(changes[layer].change, largest)),
      title: (layer) => `Layer ${layer + 1}: ${changes[layer].change === null ? (layer === 0 ? 'no earlier layer to measure from' : 'not recorded') : `${changes[layer].change.toExponential(1)} relative change`}`,
      onSelect: inspect,
    }),
    h('p', { className: 'demo-note' }, 'Layer 1 is hatched: there is no recorded layer before it to measure from.'),
    ...[actions?.project
      ? leaning(run, index, projections?.get(`${run.id}:${index}`) ?? null, actions)
      : gap({ what: 'Which words each layer leaned toward is not shown here: projecting needs the model loaded, which the Explore page does.' })].flat(),
  );
}

