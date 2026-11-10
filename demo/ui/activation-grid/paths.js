import config from './config.json' with { type: 'json' };
import { colorGrid, heatmapColor, heatmapColors, projectFrame, layerShape } from './data.js';

/** @param {readonly (number | string | undefined)[]} values */
function rms(values) {
  if (!values.length) return null;
  let sum = 0;
  for (const value of values) {
    if (value === undefined) return null;
    if (typeof value !== 'number' || !Number.isFinite(value)) return NaN;
    sum += value ** 2;
  }
  return Math.sqrt(sum / values.length);
}

/** @type {typeof import('./paths.js').modelPathView} */
export function modelPathView(recording, atMs, position, selectedLayer) {
  const trace = recording.trace;
  const captures = [...trace.frames, ...(trace.activity?.frames ?? [])].filter(f => f.atMs <= atMs
    && f.status === 'captured' && f.values && position >= f.tokenStart && position < f.tokenStart + f.tokenIds.length);
  captures.sort((a, b) => a.atMs - b.atMs);
  const last = captures[captures.length - 1];
  const lastLayer = captures.filter(f => f.layer >= 0).at(-1);
  const layer = selectedLayer >= 0 ? selectedLayer : lastLayer?.layer ?? 0;
  let index = -1;
  for (let i = 0; i < trace.frames.length; i++) if (trace.frames[i].atMs <= atMs
    && position >= trace.frames[i].tokenStart && position < trace.frames[i].tokenStart + trace.frames[i].tokenIds.length) index = i;
  const outputs = index >= 0 ? projectFrame(trace, index, position).values : new Array(trace.geometry.layers * trace.geometry.width).fill(undefined);
  const layers = Array.from({ length: trace.geometry.layers }, (_, i) => ({layer:i,
    magnitude:rms(outputs.slice(i * trace.geometry.width, (i + 1) * trace.geometry.width))}));
  const types = trace.activity?.model.layerTypes;
  const kind = Array.isArray(types) ? String(types[layer] ?? 'unknown') : 'unknown';
  const linear = kind === 'linear_attention';
  /** @type {import('./paths.js').PathNode[]} */
  const nodes = [];
  /** @param {string} id @param {string} label @param {string} role @param {number} x @param {number} y @param {number} width */
  const add = (id, label, role, x, y, width) => {
    const f = captures.filter(f => f.layer === (role === 'logits' ? -1 : layer) && f.role === role).at(-1);
    let magnitude = null, count = 0, method = 'not captured';
    if (f?.values) {
      if ('summary' in f && f.summary) {
        let sum = 0;
        for (let i = 0; i < f.summary.values.length; i++) {
          const n = Math.min(f.summary.blockSize, f.summary.observedValues - i * f.summary.blockSize);
          sum += Number(f.summary.values[i]) ** 2 * n;
        }
        count = f.summary.observedValues; magnitude = count ? Math.sqrt(sum / count) : null;
        method = `captured state summary; ${count} values; after positions ${f.tokenStart}–${f.tokenStart + f.tokenIds.length - 1}`;
      } else {
        const values = role === 'convolution-history' || role === 'recurrent-memory' ? f.values
          : f.values.slice((position - f.tokenStart) * ('fullShape' in f ? f.fullShape[1] : trace.geometry.width),
            (position - f.tokenStart + 1) * ('fullShape' in f ? f.fullShape[1] : trace.geometry.width));
        count = values.length; magnitude = rms(values); method = `RMS of ${count} captured values`;
      }
    }
    nodes.push({ id, label, role, x, y, width, magnitude, count, method,
      latest: Boolean(f && last && f.role === last.role && f.layer === last.layer && f.atMs === last.atMs) });
  };
  add('input', 'Layer input', '', 160, 30, 150);
  if (layer > 0) { nodes[0].magnitude = layers[layer - 1].magnitude; nodes[0].count = trace.geometry.width; nodes[0].method = 'Previous captured layer output; intervening transforms omitted'; }
  add('a', linear ? 'Retention input' : 'Query', linear ? 'linear-a' : 'attention-query', 54, 115, 96);
  add('b', linear ? 'Update input' : 'Key', linear ? 'linear-b' : 'kv-key', 160, 115, 96);
  add('c', linear ? 'Conv history' : 'Value', linear ? 'convolution-history' : 'kv-value', 266, 115, 96);
  add('mixer', linear ? 'Recurrent state' : 'Attention result', linear ? 'recurrent-memory' : '', 160, 180, 150);
  add('gate', 'FFN gate', 'ffn-gate', 80, 285, 132);
  add('up', 'FFN up', 'ffn-up', 240, 285, 132);
  add('product', 'Activated product', 'ffn-product', 160, 365, 150);
  add('down', 'Down input', 'ffn-down-input', 160, 435, 150);
  add('output', 'Layer output', 'layer-output', 160, 515, 150);
  if (layer === trace.geometry.layers - 1) add('scores', 'Next-token scores', 'logits', 160, 590, 150);
  return { layer, kind, layers, nodes, outputs, width: trace.geometry.width };
}

/** @type {typeof import('./paths.js').renderModelPaths} */
export function renderModelPaths(container, graph, scale, activationScale, onLayer, dark = false) {
  const svgNS = 'http://www.w3.org/2000/svg';
  /** @param {string} tag @param {Record<string,string | number>} attrs @param {string} [text] */
  const element = (tag, attrs, text) => {
    const node = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  let overview = container.querySelector('.activity-path-activations');
  if (!overview) {
    overview = document.createElement('figure'); overview.className = 'activity-path-activations';
    const caption = document.createElement('figcaption'); caption.textContent = 'Layer outputs · rows = layers, columns = hidden dimensions';
    const canvas = document.createElement('canvas'); canvas.setAttribute('aria-label', 'Live layer activations');
    const key = document.createElement('p'); key.className = 'activity-path-activation-key';
    overview.append(caption, canvas, key); container.prepend(overview);
  }
  const activationCanvas = /** @type {HTMLCanvasElement} */ (overview.querySelector('canvas'));
  activationCanvas.width = graph.width; activationCanvas.height = graph.layers.length;
  const context = activationCanvas.getContext('2d');
  if (!context) throw new Error('Layer activity requires a canvas 2D context.');
  const image = context.createImageData(graph.width, graph.layers.length);
  image.data.set(colorGrid(graph.outputs, activationScale, 'signed', dark).rgba.subarray(0, image.data.length));
  context.putImageData(image, 0, 0);
  const key = /** @type {HTMLElement} */ (overview.querySelector('.activity-path-activation-key'));
  const colors = heatmapColors('signed', dark).map(c => `rgb(${c.join(',')})`).join(',');
  key.style.backgroundImage = `linear-gradient(to right,${colors})`;
  key.textContent = `≤ −${Number(activationScale.toPrecision(3))} · 0 · ≥ ${Number(activationScale.toPrecision(3))} · fixed scale`;
  const list = container.querySelector('.activity-path-layers') ?? document.createElement('div'); list.className = 'activity-path-layers'; list.setAttribute('aria-label','Model layers');
  for (const l of graph.layers) {
    const button = /** @type {HTMLButtonElement} */ (list.children[l.layer] ?? document.createElement('button')); button.type = 'button'; button.className = 'activity-path-layer';
    button.textContent = String(l.layer + 1); button.setAttribute('aria-label',`Inspect layer ${l.layer + 1}`);
    button.setAttribute('aria-pressed', String(l.layer === graph.layer));
    button.style.setProperty('--path-color', `rgb(${(l.magnitude === null ? config.colors.missing[0] : heatmapColor(l.magnitude, scale, 'positive')).join(',')})`);
    button.title = l.magnitude === null ? 'Layer output not captured yet' : `Layer output RMS ${l.magnitude}`;
    button.onclick = () => onLayer(l.layer); if (!button.parentElement) list.append(button);
  }
  const detail = document.createElement('div'); detail.className = 'activity-path-detail';
  const heading = document.createElement('p'); heading.className = 'demo-note';
  heading.textContent = `Layer ${graph.layer + 1} · ${graph.kind === 'linear_attention' ? 'Linear memory' : graph.kind === 'full_attention' ? 'Full attention' : graph.kind}`;
  const svg = element('svg', {viewBox:graph.nodes.some(n => n.id === 'scores') ? '0 0 320 620' : '0 0 320 550',role:'img','aria-label':`Captured computation paths in layer ${graph.layer + 1}`});
  const defs = element('defs', {}), marker = element('marker', {id:'activity-path-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});
  marker.append(element('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'currentColor'})); defs.append(marker); svg.append(defs);
  // Structural connections, not measured per-edge influence or transport timing.
  for (const [path, omitted] of [
    ['M160 50 V75 H54 V95',false],['M160 50 V95',false],['M160 75 H266 V95',false],
    ['M54 135 V148 H160 V160',false],['M160 135 V160',false],['M266 135 V148 H160',false],
    ['M160 200 V240 H80 V264',true],['M160 240 H240 V264',true],
    ['M80 306 V328 H160 V344',false],['M240 306 V328 H160',false],
    ['M160 386 V414',false],['M160 456 V494',true],
  ]) svg.append(element('path', {d:String(path), fill:'none', stroke:'currentColor','stroke-width':1.3,
    'stroke-dasharray':omitted ? '3 4' : '', 'marker-end':'url(#activity-path-arrow)',opacity:.65}));
  if (graph.nodes.some(n => n.id === 'scores')) svg.append(element('path',{d:'M160 536 V569',fill:'none',stroke:'currentColor','stroke-dasharray':'3 4','marker-end':'url(#activity-path-arrow)'}));
  svg.append(element('text',{x:160,y:222,'text-anchor':'middle',class:'activity-path-note'},'Mixer output / normalization omitted'));
  svg.append(element('text',{x:160,y:478,'text-anchor':'middle',class:'activity-path-note'},'Projection / residual transforms omitted'));
  for (const n of graph.nodes) {
    const group = element('g', {'data-role':n.role, 'data-captured':n.magnitude !== null ? 'true':'false','data-latest':String(n.latest)});
    const rgb = n.magnitude === null ? config.colors.missing[0] : heatmapColor(n.magnitude, scale, 'positive');
    group.append(element('rect',{x:n.x-n.width/2,y:n.y-20,width:n.width,height:40,rx:5,class:'activity-path-node'}));
    group.append(element('rect',{x:n.x-n.width/2+2,y:n.y+13,width:n.width-4,height:5,rx:2,fill:`rgb(${rgb.join(',')})`}));
    group.append(element('text',{x:n.x,y:n.y-4,'text-anchor':'middle',class:'activity-path-label'},n.label));
    group.append(element('text',{x:n.x,y:n.y+9,'text-anchor':'middle',class:'activity-path-value'},n.magnitude === null ? 'not captured' : !Number.isFinite(n.magnitude) ? 'invalid capture' : `${Number(n.magnitude.toPrecision(3))} RMS`));
    group.append(element('title',{},`${n.label}: ${n.method}${n.latest ? ' · latest capture' : ''}`)); svg.append(group);
  }
  detail.append(heading, svg);
  container.querySelector('.activity-path-detail')?.remove();
  if (!list.parentElement) container.append(list);
  container.append(detail);
  const selected = /** @type {HTMLElement | undefined} */ (list.children[graph.layer]);
  if (selected) {
    if (list.scrollWidth > list.clientWidth && (selected.offsetLeft < list.scrollLeft || selected.offsetLeft + selected.offsetWidth > list.scrollLeft + list.clientWidth)) list.scrollLeft = selected.offsetLeft - list.clientWidth / 2;
    if (list.scrollHeight > list.clientHeight && (selected.offsetTop < list.scrollTop || selected.offsetTop + selected.offsetHeight > list.scrollTop + list.clientHeight)) list.scrollTop = selected.offsetTop - list.clientHeight / 2;
  }
}

/** @type {typeof import('./paths.js').renderLayerMap} */
export function renderLayerMap(container, options) {
  const { values, width, layers, types, scale, history, historyScale, position, difference, baseline, selected, dark, onLayer, onPosition, onInspect } = options;
  if (!container.children.length) {
    container.innerHTML = `<div class="activity-layer-overview"><div class="activity-layer-tiles" aria-label="Model layers"></div><figure class="activity-history"><figcaption>Change over tokens</figcaption><div class="activity-history-plot"><canvas tabindex="0" role="slider" aria-label="Token position in layer history"></canvas><i aria-hidden="true"></i></div><div class="activity-history-axis"></div><div class="activity-history-key"></div></figure></div><p class="activity-layer-note demo-note"></p><div class="activity-layer-detail" hidden><button type="button" class="btn btn-ghost btn-small">Close layer</button><div class="activity-paths"></div></div>`;
  }
  const tiles = /** @type {HTMLElement} */ (container.querySelector('.activity-layer-tiles'));
  tiles.style.setProperty('--layer-columns', String(Math.min(layers, config.layerColumns)));
  while (tiles.children.length > layers) tiles.lastElementChild?.remove();
  const shape = layerShape(width);
  /** @param {HTMLCanvasElement} canvas @param {(number | string | undefined)[]} scalars @param {number} w @param {number} h @param {number} limit @param {'signed'|'positive'} range */
  const draw = (canvas, scalars, w, h, limit, range) => {
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Layer maps require canvas 2D.');
    const image = ctx.createImageData(w,h);
    image.data.set(colorGrid(scalars, limit, range, dark).rgba.subarray(0,w*h*4));
    ctx.putImageData(image,0,0);
  };
  for (let l = 0; l < layers; l++) {
    let tile = /** @type {HTMLButtonElement | undefined} */ (tiles.children[l]);
    if (!tile) {
      tile = document.createElement('button'); tile.type = 'button'; tile.className = 'activity-layer-tile';
      tile.innerHTML = '<span></span><canvas></canvas><small></small>'; tiles.append(tile);
    }
    const label = types?.[l] === 'linear_attention' ? 'Linear' : types?.[l] === 'full_attention' ? 'Attention' : 'Layer output';
    tile.dataset.layer = String(l); tile.dataset.kind = types?.[l] ?? 'unknown';
    tile.disabled = !['linear_attention','full_attention'].includes(types?.[l] ?? '');
    tile.setAttribute('aria-label', `Layer ${l+1}, ${label}. ${tile.disabled ? 'Branch structure not recorded.' : 'Inspect captured branches.'}`);
    tile.setAttribute('aria-pressed', String(selected === l));
    /** @type {HTMLElement} */ (tile.querySelector('span')).textContent = `${l+1}${l < layers-1 ? ' →' : ''}`;
    /** @type {HTMLElement} */ (tile.querySelector('small')).textContent = label;
    tile.onclick = () => onLayer(selected === l ? -1 : l);
    const canvas = /** @type {HTMLCanvasElement} */ (tile.querySelector('canvas'));
    const row = values.slice(l*width,(l+1)*width);
    draw(canvas, row, shape.width, shape.height, scale, 'signed');
    canvas.onpointermove = event => {
      const rect = canvas.getBoundingClientRect();
      const d = Math.floor((event.clientY-rect.top)/rect.height*shape.height)*shape.width + Math.floor((event.clientX-rect.left)/rect.width*shape.width);
      onInspect(`Layer ${l+1} · component ${d} · ${row[d] ?? 'missing'}${difference ? ' change' : ''}`);
    };
    canvas.onpointerleave = () => onInspect(null);
  }
  const figure = /** @type {HTMLElement} */ (container.querySelector('.activity-history'));
  /** @type {HTMLElement} */ (figure.querySelector('figcaption')).textContent = difference ? 'Change over tokens' : 'Activity over tokens';
  const canvas = /** @type {HTMLCanvasElement} */ (figure.querySelector('canvas'));
  const n = history.positions.length;
  draw(canvas, history.values, Math.max(1,n), layers, historyScale, 'positive');
  // Tokens after the one being shown are not displayed yet; that is not missing data.
  const firstLater = history.positions.findIndex(p => p > position);
  const historyContext = canvas.getContext('2d');
  if (firstLater >= 0 && historyContext) {
    historyContext.fillStyle = `rgb(${config.colors.padding.join(',')})`;
    historyContext.fillRect(firstLater, 0, n - firstLater, layers);
  }
  canvas.setAttribute('aria-valuemin', String(history.positions[0] ?? 0));
  canvas.setAttribute('aria-valuemax', String(history.positions[n-1] ?? 0));
  canvas.setAttribute('aria-valuenow', String(position));
  canvas.setAttribute('aria-valuetext', `Token ${position}. Rows are layers; color is ${difference ? 'RMS change' : 'RMS activity'}. Arrow keys select a token.`);
  const column = Math.max(0,history.positions.indexOf(position));
  const cursor = /** @type {HTMLElement} */ (figure.querySelector('i'));
  cursor.style.left = `${(column+.5)/Math.max(1,n)*100}%`;
  const key = /** @type {HTMLElement} */ (figure.querySelector('.activity-history-key'));
  key.style.borderImage = `linear-gradient(to right,${heatmapColors('positive',dark).map(c=>`rgb(${c.join(',')})`).join(',')}) 1`;
  key.textContent = `RMS ${difference ? 'change' : 'activity'} · 0–${Number(historyScale.toPrecision(3))}`;
  /** @type {HTMLElement} */ (figure.querySelector('.activity-history-axis')).textContent = `Tokens ${history.positions[0] ?? 0}–${history.positions[n-1] ?? 0} → · layers 1–${layers} ↓${firstLater >= 0 ? ` · after token ${position}: not shown yet` : ''}`;
  canvas.onclick = event => {
    const box = canvas.getBoundingClientRect();
    const c = Math.max(0,Math.min(n-1,Math.floor((event.clientX-box.left)/box.width*n)));
    if (n) onPosition(history.positions[c]);
  };
  canvas.onkeydown = event => {
    const c = event.key === 'Home' ? 0 : event.key === 'End' ? n-1 : event.key === 'ArrowLeft' ? column-1 : event.key === 'ArrowRight' ? column+1 : null;
    if (c !== null && n) { event.preventDefault(); onPosition(history.positions[Math.max(0,Math.min(n-1,c))]); }
  };
  canvas.onpointermove = event => {
    const box = canvas.getBoundingClientRect();
    const c = Math.min(n-1,Math.floor((event.clientX-box.left)/box.width*n)), l = Math.min(layers-1,Math.floor((event.clientY-box.top)/box.height*layers));
    onInspect(`Layer ${l+1} · token ${history.positions[c]} · RMS ${difference ? 'change' : 'activity'}: ${history.positions[c] > position ? 'not shown yet' : history.values[l*n+c] ?? 'missing'}`);
  };
  canvas.onpointerleave = () => onInspect(null);
  /** @type {HTMLElement} */ (container.querySelector('.activity-layer-note')).textContent = `${width.toLocaleString()} values/layer · ${shape.width} × ${shape.height} tile · ${baseline ? 'compared with saved run' : difference ? 'change from previous position' : 'captured outputs'}`;
  const detail = /** @type {HTMLElement} */ (container.querySelector('.activity-layer-detail'));
  detail.hidden = selected < 0;
  /** @type {HTMLButtonElement} */ (detail.querySelector('button')).onclick = () => onLayer(-1);
}
