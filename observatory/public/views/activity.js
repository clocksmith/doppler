// Activity: the pixel grids, drawn for every run that recorded layer outputs
// (each live answer does unless the menu turns them off). An overview heatmap
// shows, for every token position and layer, how far that layer's output moved
// from the previous position; below it, one tile per layer shows the selected
// position's values, one pixel per value. Selecting a cell or tile selects that
// position and layer everywhere.

import { layerVector } from '../shared-run.js';
import { $, h } from '../viz/dom.js';
import { gap } from '../viz/gap.js';
import { provenanceMark } from '../viz/provenance.js';

const modes = new Map(); // run id -> 'change' | 'value'

// Side of the square tile that holds `width` values (the last row may be short).
export const tileSide = (width) => Math.ceil(Math.sqrt(width));

// A diverging colour for v in [-1, 1]: blue below zero, red above, white at zero.
export function diverging(v) {
  const t = Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0));
  const [r, g, b] = t < 0 ? [37, 99, 235] : [209, 59, 77];
  const k = Math.abs(t);
  return [Math.round(255 + (r - 255) * k), Math.round(255 + (g - 255) * k), Math.round(255 + (b - 255) * k)];
}

// The value below which `share` of |values| fall: a scale that one outlier cannot flatten.
export function robustScale(values, share = 0.99) {
  const magnitudes = Array.from(values, Math.abs).filter(Number.isFinite).sort((a, b) => a - b);
  if (!magnitudes.length) return null;
  return magnitudes[Math.min(magnitudes.length - 1, Math.floor(share * (magnitudes.length - 1)))] || magnitudes.at(-1) || null;
}

// The vector a tile shows: the layer's output, or how it moved from the previous recorded position.
function vectorFor(run, positions, p, layer, mode) {
  const now = layerVector(run, positions[p], layer);
  if (!now || mode === 'value') return now;
  const before = p > 0 ? layerVector(run, positions[p - 1], layer) : null;
  if (!before) return null;
  return now.map((value, i) => value - before[i]);
}

const rms = (vector) => (vector ? Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0) / vector.length) : null);

function canvas(width, height, className, label) {
  const node = h('canvas', { className, width, height, attrs: { role: 'img', 'aria-label': label } });
  return { node, context: node.getContext?.('2d') ?? null };
}

export function renderActivity({ run, selection }) {
  const root = $('activity');
  if (!root) return;
  const saved = run.layerOutputs;
  if (!saved) {
    root.replaceChildren(gap({ what: 'This run was recorded without layer outputs, so there are no grids. Turn on "Layer outputs" in the Model menu and ask again.' }));
    return;
  }
  const mode = modes.get(run.id) ?? 'change';
  const { positions, layers, width } = saved;
  const chosen = selection.get();
  const p = chosen.position === null ? -1 : positions.indexOf(chosen.position);
  const prompt = run.inputTokenIds?.length ?? 0;
  const label = (position) => { const k = position - prompt; return k >= 0 && run.tokens[k] ? `word ${k + 1} "${run.tokens[k].text ?? ''}"` : `prompt position ${position}`; };

  // Overview: one cell per (position, layer), the RMS of the tile it opens.
  const cells = positions.map((_, i) => Array.from({ length: layers }, (_, layer) => rms(vectorFor(run, positions, i, layer, mode))));
  // Each layer (row) on its own scale: later layers' outputs are far larger, and one
  // shared scale would leave the early layers blank.
  const rowScale = Array.from({ length: layers }, (_, layer) => robustScale(cells.map((column) => column[layer]).filter((v) => v !== null)) ?? 1);
  const overview = canvas(positions.length, layers, 'ws-act-overview', `${mode === 'change' ? 'Change from the previous position' : 'Size of the layer output'}, by position and layer`);
  if (overview.context) {
    const image = overview.context.createImageData(positions.length, layers);
    cells.forEach((column, x) => column.forEach((value, layer) => {
      const at = (layer * positions.length + x) * 4;
      if (value === null) { image.data.set([200, 200, 200, 255], at); return; }
      const t = Math.min(1, value / rowScale[layer]);
      // Dark navy to yellow, like the demo's change map.
      image.data.set([Math.round(20 + 235 * t * t), Math.round(30 + 200 * t), Math.round(90 + 20 * (1 - t)), 255], at);
    }));
    overview.context.putImageData(image, 0, 0);
  }
  overview.node.addEventListener('click', (event) => {
    const box = overview.node.getBoundingClientRect();
    const x = Math.floor(((event.clientX - box.left) / box.width) * positions.length);
    const layer = Math.floor(((event.clientY - box.top) / box.height) * layers);
    if (positions[x] !== undefined && layer >= 0 && layer < layers) selection.select({ position: positions[x], layer });
  });
  const marker = p >= 0 ? h('div', { className: 'ws-act-marker', style: `left: ${((p + 0.5) / positions.length) * 100}%` }) : null;

  // Tiles: every layer at the selected position.
  let tiles = h('p', { className: 'demo-note' }, 'Select a word, or a column in the map above, to see its layers as tiles.');
  let scaleNote = '';
  if (p >= 0) {
    const vectors = Array.from({ length: layers }, (_, layer) => vectorFor(run, positions, p, layer, mode));
    // Each tile on its own scale, for the same reason as the map's rows; the scale is in its tooltip.
    const scales = vectors.map((vector) => (vector ? robustScale(vector) ?? 1 : null));
    scaleNote = 'Each tile has its own colour scale (99th percentile of |value| in that layer, shown on hover); values beyond it are clipped.';
    const side = tileSide(width);
    tiles = h('div', { className: 'ws-act-tiles' }, ...vectors.map((vector, layer) => {
      const tile = canvas(side, side, `ws-act-tile${chosen.layer === layer ? ' is-selected' : ''}`, `Layer ${layer + 1}`);
      if (vector && tile.context) {
        const image = tile.context.createImageData(side, side);
        for (let i = 0; i < side * side; i++) {
          const [r, g, b] = i < vector.length ? diverging(vector[i] / scales[layer]) : [128, 128, 128];
          image.data.set([r, g, b, i < vector.length ? 255 : 40], i * 4);
        }
        tile.context.putImageData(image, 0, 0);
      }
      return h('button', { type: 'button', className: 'ws-act-cell', title: vector ? `Layer ${layer + 1} · scale ±${scales[layer].toPrecision(3)}` : `Layer ${layer + 1}: not recorded`, on: { click: () => selection.select({ layer }) } },
        vector ? tile.node : h('span', { className: 'ws-act-missing' }), h('span', { className: 'ws-act-label' }, String(layer + 1)));
    }));
  }

  const switcher = h('div', { className: 'ws-act-modes', attrs: { role: 'group', 'aria-label': 'What the grids show' } },
    ...[['change', 'Change from previous token'], ['value', 'Layer output']].map(([id, text]) => h('button', {
      type: 'button', className: 'ws-tab', ariaPressed: String(mode === id), attrs: { 'aria-selected': String(mode === id) },
      on: { click: () => { modes.set(run.id, id); renderActivity({ run, selection }); } },
    }, text)));
  root.replaceChildren(
    h('p', {}, provenanceMark(mode === 'change' ? 'derived' : 'measured'), mode === 'change'
      ? ' How far each layer\'s output moved from the previous token, recomputed from the recorded outputs.'
      : ' Each layer\'s recorded output (the hidden state after that layer).'),
    switcher,
    h('div', { className: 'ws-act-overview-wrap' }, h('div', { className: 'ws-act-axis demo-note' }, `${layers} layers ↓`), h('div', { className: 'ws-act-map' }, overview.node, marker)),
    h('p', { className: 'demo-note' }, `Positions ${positions[0]}–${positions.at(-1)} → (prompt and answer). Each layer's row on its own scale: darker is less, yellow is more for that layer; grey was not recorded. Select a cell for that position and layer.`),
    h('h3', { className: 'ws-act-title' }, p >= 0 ? `${label(positions[p])}: ${layers} layers, ${width} values each` : 'Layers'),
    tiles,
    ...(scaleNote ? [h('p', { className: 'demo-note' }, `${scaleNote} Blue below zero, red above. One pixel per value, ${tileSide(width)}×${tileSide(width)} per layer.`)] : []),
  );
}
