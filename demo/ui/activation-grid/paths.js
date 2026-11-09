import config from './config.json' with { type: 'json' };
import { heatmapColor, projectFrame } from './data.js';

/** @param {(number | string | undefined)[]} values */
function rms(values) {
  if (!values.length) return null;
  if (values.some(v => typeof v !== 'number' || !Number.isFinite(v))) return NaN;
  let sum = 0;
  for (const value of values) sum += Number(value) ** 2;
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
  return { layer, kind, layers, nodes };
}

/** @type {typeof import('./paths.js').renderModelPaths} */
export function renderModelPaths(container, graph, scale, onLayer) {
  const svgNS = 'http://www.w3.org/2000/svg';
  /** @param {string} tag @param {Record<string,string | number>} attrs @param {string} [text] */
  const element = (tag, attrs, text) => {
    const node = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
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
