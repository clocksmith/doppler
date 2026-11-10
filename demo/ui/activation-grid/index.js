import config from './config.json' with { type: 'json' };
import previews from './previews.json' with { type: 'json' };
import previewData from '../../assets/activity-previews.json' with { type: 'json' };
import { SHELL_MANIFEST_DIGEST } from '../../generated-shell-manifest.js';
import { validateRecording, colorGrid, initialColorScale, heatmapColor, heatmapColors, layerHistory, layerShape } from './data.js';
import { modelPathView, renderModelPaths, renderLayerMap } from './paths.js';
import { encodeRecording } from './recording-codec.js';
import { activityView, activityModes, playbackPoints, assertMatchedRuns, missingReason } from './views.js';

/** @template {keyof import('./index.js').GridElements} K
 * @param {K} id @returns {import('./index.js').GridElements[K]} */
function $(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Activation grid is missing ${id}.`);
  return /** @type {import('./index.js').GridElements[K]} */ (node);
}
/** @type {import('./data.js').ActivationRecording | null} */
let recording = null;
let cursor = 0, follow = true;
/** @type {ReturnType<typeof setTimeout> | null} */
let timer = null;
/** @type {number | null} */
let scheduled = null;
let active = false;
/** @type {import('./data.js').ActivationRecording | null} */
let baseline = null;
let revision = 0;
let exampleLoading = false;
let exampleError = '';
/** @type {AbortController | null} */
let exampleRequest = null;
let source = 'Saved recording';
let currentPrompt = '';
/** @type {((recording: import('./data.js').ActivationRecording, options?: { preserveDraft?: boolean }) => void) | null} */
let onOpenRecording = null;
/** @type {(() => { position: number; probability: number }[]) | null} */
let unsureWords = null;
/** @type {((position: number) => void) | null} */
let onUnsureWord = null;
// True while a first-visit example sits paused where it opened.
let openingHint = false;
/** @type {((recording: import('./data.js').ActivationRecording, point: import('./views.js').PlaybackPoint) => void) | null} */
let onPlayback = null;

export function isActivationRecordingEnabled() { return $('activation-record')?.checked === true; }

function stopPlayback() {
  if (timer !== null) clearTimeout(timer);
  timer = null;
  if ($('activation-play')) $('activation-play').textContent = 'Play';
}

function scaleKey() { return `${$('activity-mode').value}:${$('activity-detail').value}`; }

function points() { return recording ? playbackPoints(recording, $('activity-playback').value, $('activity-mode').value) : []; }

// A change needs an earlier token: without a saved run, play starts at the second.
function firstPlayable() {
  return $('activity-mode').value === 'difference' && !baseline && $('activity-playback').value === 'tokens' && points().length > 1 ? 1 : 0;
}

/** @param {{ position: number }[]} timeline @param {number} position */
function nearestIndex(timeline, position) {
  let best = 0;
  for (let i = 1; i < timeline.length; i++) {
    if (Math.abs(timeline[i].position - position) < Math.abs(timeline[best].position - position)) best = i;
  }
  return best;
}

// With a recording open: only views it can draw, "Start here" first; the rest under Details.
/** @param {ReturnType<typeof activityModes>} availability */
function syncViewList(availability) {
  const select = $('activity-mode'), open = Boolean(recording?.trace.frames.length);
  const listed = availability.filter(m => !open || !m.reason);
  const signature = listed.map(m => m.id).join(',');
  if (select.dataset.signature !== signature) {
    const previous = select.value || config.defaultMode;
    const common = config.commonModes.map(id => listed.find(m => m.id === id)).filter(m => m !== undefined);
    const groups = [{ name: 'Start here', items: common }, ...[...new Set(listed.map(m => m.group))]
      .map(name => ({ name, items: listed.filter(m => m.group === name && !common.includes(m)) }))];
    select.replaceChildren();
    for (const { name, items } of groups) {
      if (!items.length) continue;
      const group = document.createElement('optgroup'); group.label = name;
      for (const mode of items) group.append(new Option(mode.label, mode.id));
      select.append(group);
    }
    select.dataset.signature = signature;
    const kept = listed.find(m => m.id === previous) ?? listed.find(m => m.id === config.defaultMode) ?? listed[0];
    select.value = kept?.id ?? '';
    // A view that left the list hands over to one with its own clock.
    if (kept && kept.id !== previous) $('activity-playback').value = kept.clock === 'boundary' ? 'captures' : 'tokens';
  }
  for (const option of select.options) {
    const mode = availability.find(m => m.id === option.value);
    option.disabled = Boolean(mode?.reason);
    option.title = mode?.reason ?? mode?.label ?? option.value;
    option.textContent = `${mode?.label}${mode?.reason ? ' · unavailable' : ''}`;
  }
  const unlisted = open ? availability.filter(m => m.reason) : [];
  $('activity-unlisted').hidden = !unlisted.length;
  $('activity-unlisted').textContent = unlisted.length
    ? `Not in this recording: ${unlisted.map(m => `${m.label} (${m.reason})`).join('; ')}` : '';
}

/** Timeline indexes of the words the model gave less than even odds. */
function unsureIndexes(/** @type {{ position: number }[]} */ timeline) {
  const unsure = new Set((unsureWords?.() ?? []).map(word => word.position));
  return timeline.flatMap((point, index) => unsure.has(point.position) ? [index] : []);
}

/** @param {1 | -1} direction */
function jumpToUnsure(direction) {
  const timeline = points(), indexes = unsureIndexes(timeline);
  const target = direction > 0 ? indexes.find(i => i > cursor) : [...indexes].reverse().find(i => i < cursor);
  if (target === undefined) return;
  stopPlayback(); follow = false; openingHint = false; cursor = target;
  $('activation-position').value = String(timeline[cursor].position);
  // Painting brings the answer to this word; then its alternatives can be shown.
  paint();
  onUnsureWord?.(timeline[cursor].position);
}

function paint() {
  scheduled = null;
  const dark = globalThis.matchMedia?.('(prefers-color-scheme: dark)').matches === true;
  const availability = activityModes(recording);
  syncViewList(availability);
  const trace = recording?.trace, timeline = points();
  $('activation-live').setAttribute('aria-pressed', String(active && follow));
  $('activation-live').disabled = !active;
  $('activation-live').hidden = !active || follow;
  $('activation-play').hidden = active && follow;
  $('activity-gallery').classList.toggle('activity-gallery--compact', timeline.length > 0 || active);
  $('activity-intro').hidden = timeline.length > 0 || active;
  $('activity-intro').textContent = isActivationRecordingEnabled()
    ? 'Activity will appear as your answer is generated.'
    : 'Activity recording is off.';
  $('activity-selectors').hidden = !timeline.length;
  $('activation-caption').hidden = !timeline.length;
  $('activity-source').hidden = !exampleLoading && (!timeline.length || source === 'Current run' || source === 'Recorded replay');
  if (!exampleLoading && recording) $('activity-source').textContent = source;
  const select = $('activity-mode');
  const chosen = availability.find(m => m.id === select.value), snapshot = chosen?.clock === 'snapshot';
  $('activation-timeline').hidden = snapshot;
  const unsure = snapshot || active ? [] : unsureIndexes(timeline);
  $('activity-unsure').hidden = !unsure.length;
  $('activity-unsure-prev').disabled = !unsure.some(i => i < cursor);
  $('activity-unsure-next').disabled = !unsure.some(i => i > cursor);
  $('activity-hint').hidden = !openingHint || !timeline.length;
  for (const id of ['activity-playback', 'activation-speed', 'activation-position']) {
    const element = $(/** @type {'activity-playback'} */ (id));
    const field = element.closest('label') ?? element.closest('.demo-range-field');
    if (field) field.hidden = snapshot;
  }
  $('activity-playback').disabled = chosen?.clock === 'boundary';
  for (const card of $('activity-gallery').querySelectorAll('button')) {
    card.setAttribute('aria-pressed', String(card.dataset.mode === select.value && timeline.length > 0));
    card.disabled = exampleLoading || (Boolean(recording) && Boolean(availability.find(m => m.id === card.dataset.mode)?.reason));
  }
  $('activity-availability').textContent = exampleError || (timeline.length ? chosen?.reason ?? '' : '');
  $('activity-availability').hidden = !$('activity-availability').textContent;
  $('activity-paths-button').disabled = Boolean(availability.find(m => m.id === 'model-paths')?.reason);
  $('activity-weight-controls').hidden = !['weights', 'contributions', 'residency', 'quantization-error', 'recurrent-memory', 'convolution-history'].includes(select.value);
  const weightLabel = $('activity-weight').closest('label');
  if (weightLabel) weightLabel.hidden = ['recurrent-memory', 'convolution-history'].includes(select.value);
  $('activity-compare-controls').hidden = select.value !== 'difference';
  $('activation-export').disabled = !timeline.length;
  $('activation-play').disabled = !timeline.length;
  $('activation-scrub').disabled = !timeline.length;
  $('activation-player').hidden = !timeline.length;
  $('activation-detail-controls').hidden = !timeline.length;
  if (!timeline.length || !recording || !trace) {
    $('activity-go').hidden = true;
    $('activation-canvas').width = 1; $('activation-canvas').height = 1;
    $('activation-layout').textContent = ''; $('activation-status').textContent = '';
    $('activation-caption').textContent = '';
    return;
  }
  const layerSelect = $('activity-layer');
  if (layerSelect.options.length !== trace.geometry.layers + 1) {
    layerSelect.replaceChildren(new Option('All captured layers', '-1'));
    for (let i = 0; i < trace.geometry.layers; i++) layerSelect.add(new Option(`Layer ${i + 1}`, String(i)));
  }
  const weightSelect = $('activity-weight'), weights = trace.activity?.weights ?? [];
  if (weightSelect.options.length !== weights.length + 1) {
    const previous = weightSelect.value;
    weightSelect.replaceChildren(new Option('All observed tensors', ''));
    for (const w of weights) weightSelect.add(new Option(`${w.id} · ${w.shape.join(' × ') || 'shape unavailable'}`, w.id));
    weightSelect.value = previous;
  }
  cursor = follow ? timeline.length - 1 : Math.min(cursor, timeline.length - 1);
  const point = timeline[cursor], scrub = $('activation-scrub'), token = $('activation-position');
  scrub.max = String(timeline.length - 1); scrub.value = String(cursor);
  token.min = '0'; token.max = String(Math.max(...trace.frames.map(f => f.tokenStart + f.tokenIds.length - 1)));
  if (follow || Number(token.value) > Number(token.max)) token.value = String(point.position);
  const position = Number(token.value);
  if (!active && !snapshot) onPlayback?.(recording, { ...point, position });
  const graph = select.value === 'model-paths' ? modelPathView(recording, point.atMs, position, Number(layerSelect.value)) : null;
  const tiled = select.value === 'difference' || select.value === 'layer-output';
  const view = graph ? { values: graph.nodes.map(n => n.magnitude ?? undefined), range: /** @type {const} */ ('positive'),
    representation:'aggregated', quantity:'Captured RMS magnitude', valuesPerScalar:1, underlyingValues:0, extent:graph.nodes.length, offset:0, segments:[],
    description:'Color shows captured magnitude, not importance. Connections show computational structure; dashed links omit unrecorded transformations. A bold outline marks the latest captured stage.' }
    : activityView(recording, { mode: select.value, atMs: snapshot ? Infinity : point.atMs, position,
    layer: tiled ? -1 : Number(layerSelect.value), weightId: weightSelect.value, detail: $('activity-detail').value,
    page: Number($('activity-page').value), baseline });
  const scales = recording.display.scales ??= {};
  const key = scaleKey();
  const fixedUnit = view.range === 'unit' || view.range === 'residency';
  if (fixedUnit) scales[key] = 1;
  if (view.values.some(v => typeof v === 'number' && v !== 0) && scales[key] === undefined) {
    scales[key] = initialColorScale(view.values, recording.display.scale);
  }
  const scale = fixedUnit ? 1 : scales[key] ?? recording.display.scale;
  $('activation-scale').disabled = fixedUnit;
  $('activation-scale').value = String(scale);
  const { side, rgba } = colorGrid(view.values, scale, view.range, dark);
  const canvas = $('activation-canvas'); canvas.hidden = tiled || Boolean(graph) || side === 0;
  $('activity-layer-map').hidden = !tiled;
  $('activity-paths').hidden = !graph;
  $('activity-paths-button').hidden = !graph || layerSelect.value === '-1';
  if (graph) {
    const activationKey = 'model-path-activations';
    if (scales[activationKey] === undefined && graph.outputs.some(v => typeof v === 'number' && v !== 0)) {
      scales[activationKey] = initialColorScale(graph.outputs, recording.display.scale);
    }
    renderModelPaths($('activity-paths'), graph, scale, scales[activationKey] ?? recording.display.scale,
      layer => { layerSelect.value = String(layer); queuePaint(); }, dark);
  }
  layerSelect.options[0].textContent = graph ? 'Follow latest layer' : 'All captured layers'; canvas.width = Math.max(1, side); canvas.height = Math.max(1, side);
  canvas.setAttribute('aria-label', `${view.quantity}: one scalar per opaque heatmap pixel, ${view.representation}`);
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Activity grids require a canvas 2D context.');
  if (side) { const image = context.createImageData(side, side); image.data.set(rgba); context.putImageData(image, 0, 0); }
  if (tiled) {
    const difference = select.value === 'difference';
    const history = layerHistory(recording, point.atMs, position, difference, baseline);
    const historyKey = `history:${difference}:${Boolean(baseline)}`;
    if (scales[historyKey] === undefined && history.values.some(v => typeof v === 'number' && v !== 0)) {
      scales[historyKey] = initialColorScale(history.values, recording.display.scale);
    }
    const types = trace.activity?.model.layerTypes;
    const onLayer = (/** @type {number} */ layer) => { layerSelect.value = String(layer); queuePaint(); };
    renderLayerMap($('activity-layer-map'), { values:view.values, width:trace.geometry.width, layers:trace.geometry.layers,
      types:Array.isArray(types) ? types : undefined, scale, history, historyScale:scales[historyKey] ?? recording.display.scale,
      position, difference, baseline:difference && Boolean(baseline), selected:Number(layerSelect.value), dark, onLayer,
      onPosition: pos => { stopPlayback(); follow = false; for (let i=0;i<timeline.length;i++) if (timeline[i].position === pos) cursor=i; token.value=String(pos); queuePaint(); },
      onInspect: text => { $('activation-cell').hidden = text === null; $('activation-cell').textContent = text ?? ''; } });
    if (Number(layerSelect.value) >= 0) {
      const detail = modelPathView(recording, point.atMs, position, Number(layerSelect.value));
      const magnitudes = detail.nodes.map(n => n.magnitude ?? undefined);
      if (scales['layer-stages'] === undefined && magnitudes.some(v => typeof v === 'number' && v !== 0)) scales['layer-stages'] = initialColorScale(magnitudes, recording.display.scale);
      renderModelPaths(/** @type {HTMLElement} */ ($('activity-layer-map').querySelector('.activity-layer-detail .activity-paths')),
        detail, scales['layer-stages'] ?? recording.display.scale, scale, onLayer, dark);
    }
  }
  const present = view.values.filter(v => v !== undefined).length;
  // Why values are missing, read from the recording, with the nearest token that has them.
  const why = graph || chosen?.reason ? null : missingReason(recording, select.value, position, $('activity-detail').value);
  const firstPosition = Math.min(...trace.frames.map(f => f.tokenStart));
  let missingText = '';
  if (!present && !chosen?.reason) {
    missingText = select.value === 'difference' && !baseline && position === firstPosition
      ? 'A change needs an earlier token to compare with, and this is the first one.'
      : why?.text || (layerSelect.value !== '-1'
        ? `Nothing was saved for layer ${Number(layerSelect.value) + 1} in this view at token ${position}. Choose All captured layers to see which layers have values.`
        : `Nothing was saved for this view at token ${position}.`);
  } else if (why && (why.kind === 'dropped' || why.kind === 'failed') && present < view.values.length) {
    missingText = `Part of this picture is missing. ${why.text}`;
  }
  const goTo = !present && select.value === 'difference' && !baseline && position === firstPosition ? firstPosition + 1
    : !present && why && why.nearest !== null && why.nearest !== position ? why.nearest : null;
  if (missingText) { $('activity-availability').hidden = false; $('activity-availability').textContent = missingText; }
  $('activity-go').hidden = goTo === null;
  $('activity-go').textContent = goTo === null ? '' : `Go to token ${goTo}`;
  $('activity-go').dataset.position = goTo === null ? '' : String(goTo);
  const playMeaning = snapshot ? 'Snapshot: no token playback.' : $('activity-playback').value === 'tokens'
    ? 'Play selects token positions; prefill positions were computed together.'
    : $('activity-playback').value === 'time' ? 'Play follows recorded capture timestamps, including diagnostic overhead.'
      : 'Play advances operator captures at 12 captures/second at 1×.';
  $('activation-caption').dataset.live = String(active && follow);
  $('activation-caption').textContent = `${snapshot ? chosen.clockLabel : `${point.phase === 'prefill' ? 'Prefill' : 'Decode'} · token ${position}`}${graph ? ` · layer ${graph.layer + 1}` : ''}`;
  $('activity-title').textContent = chosen?.label ?? view.quantity;
  $('activity-update').textContent = chosen?.clockLabel ?? '';
  $('activity-method').textContent = graph ? 'Stage summaries' : view.representation === 'aggregated' ? 'Block summary' : view.representation === 'raw' ? 'Captured values' : 'Reconstructed';
  $('activity-explanation').textContent = graph ? view.description : $('activity-detail').value === 'raw' || select.value === 'contributions' ? view.description : previews.find(p => p.mode === select.value)?.help ?? view.description;
  $('activity-coverage').textContent = `${present.toLocaleString()}/${view.values.length.toLocaleString()} pixels captured · ${view.valuesPerScalar === 1 ? '1 value/pixel' : `up to ${view.valuesPerScalar.toLocaleString()} values/pixel`}${select.value === 'weights' ? ' · fixed weights' : ''}`;
  $('activation-cell').hidden = true;
  if (graph) {
    $('activity-coverage').textContent = `${present}/${graph.nodes.length} stages captured · partial graph · bold outline: latest capture`;
  }
  if (select.value === 'weights' && $('activity-detail').value === 'overview') {
    $('activity-coverage').textContent += ` · ${weights.filter(w => w.rms.length).length}/${weights.length} tensors captured; ${weights.filter(w => !w.shape.length).length} shapes unknown and excluded`;
  }
  const legend = $('activity-legend'); legend.hidden = side === 0; legend.replaceChildren();
  const gradient = document.createElement('div'); gradient.className = 'activity-color-bar';
  const colors = heatmapColors(view.range, dark);
  gradient.style.background = `linear-gradient(to right, ${colors.map(c => `rgb(${c.join(',')})`).join(',')})`;
  const labels = document.createElement('div'); labels.className = 'activity-color-labels';
  const fmt = (/** @type {number} */ n) => String(Number(n.toPrecision(3)));
  for (const text of [view.range === 'signed' ? `≤ −${fmt(scale)}` : '0', view.range === 'signed' ? '0' : fmt(scale / 2), `≥ ${fmt(scale)}`]) {
    const label = document.createElement('span'); label.textContent = text; labels.append(label);
  }
  const keyNote = document.createElement('p'); keyNote.className = 'demo-note';
  const derivation = graph ? '' : view.representation === 'reconstructed'
    ? (['attention-query', 'attention-prefill', 'linear-gates'].includes(select.value) ? ' · recomputed from saved values, not a causal explanation' : ' · recomputed from saved values') : '';
  const darkMeaning = view.range === 'unit' ? ' · dark is a real value near zero; missing is checkered' : '';
  keyNote.textContent = `${view.quantity}${derivation} · fixed scale${darkMeaning}`;
  const markers = document.createElement('div'); markers.className = 'activity-color-markers';
  const markerKinds = graph ? [['Not captured', config.colors.missing[0]], ['Invalid', config.colors.invalid]]
    : [['Missing', config.colors.missing[1]], ['Invalid', config.colors.invalid], ['Padding', config.colors.padding]];
  for (const [name, color] of markerKinds) {
    const marker = document.createElement('span'), swatch = document.createElement('i');
    swatch.style.background = `rgb(${/** @type {number[]} */ (color).join(',')})`; if (name === 'Missing') swatch.style.background = `repeating-conic-gradient(rgb(${config.colors.missing[0].join(',')}) 0% 25%, rgb(${config.colors.missing[1].join(',')}) 0% 50%) 0 / 6px 6px`;
    marker.append(swatch, document.createTextNode(String(name))); markers.append(marker);
  }
  if (view.range === 'residency') {
    keyNote.textContent = '0: not resident · 0.25: resident · 1: observed dispatch. No measure of importance.';
    for (const n of [0, 0.25, 1]) { const label = document.createElement('span'); label.textContent = String(n); label.style.borderBottom = `4px solid rgb(${heatmapColor(n, 1, 'residency').join(',')})`; markers.append(label); }
  }
  legend.append(gradient, labels, keyNote, markers);
  $('activation-layout').textContent = `${present.toLocaleString()}/${view.values.length.toLocaleString()} displayed scalars present; ${view.valuesPerScalar} underlying values per pixel. Full-detail extent ${view.underlyingValues.toLocaleString()} scalars (${Math.ceil(Math.sqrt(view.underlyingValues)).toLocaleString()}² pixels); current overview/window extent ${view.extent.toLocaleString()} scalars; tile starts at ${view.offset.toLocaleString()}. ${view.description} ${playMeaning} Colors clip outside the displayed bounds; saved numerical values are unchanged.`;
  if (graph) $('activation-layout').textContent = `${trace.geometry.layers} recorded layers; ${graph.kind} at layer ${graph.layer + 1}. Node colors summarize captured values with RMS. This is a partial computation diagram: normalization, residual transforms and uncaptured operator outputs are omitted, never invented. ${playMeaning}`;
  $('activation-layout').textContent += ' Color uses 24-bit RGB, fully opaque.';
  if (tiled) {
    const shape = layerShape(trace.geometry.width);
    $('activation-layout').textContent = `${trace.geometry.layers} layer tiles, each ${shape.width} × ${shape.height}, in layer order. One component per opaque pixel; folding a vector does not imply connections between neighboring pixels. ${view.description} History: RMS across every component in a layer; any missing component makes its summary missing. Future positions stay missing. Select a tile for captured branches. ${playMeaning}`;
  }
  const gaps = [...trace.frames, ...(trace.activity?.frames ?? [])].filter(f => ['failed', 'dropped'].includes(f.status)).length;
  $('activation-status').textContent = `${trace.identity.logicalModelId ?? trace.identity.modelId} · ${trace.identity.adapters?.length ? JSON.stringify(trace.identity.adapters) : 'no active adapters recorded'} · manifest ${trace.identity.manifestHash ?? 'unavailable'} · ${trace.status} · ${gaps} missing captures · ${((trace.recordedBytes + (trace.activity?.recordedBytes ?? 0)) / 1048576).toFixed(1)} MiB of retained numerical payload, plus weights and metadata${trace.stoppedReason ? ' · ' + trace.stoppedReason : ''}${trace.activity?.unavailable['capture-limit'] ? ' · ' + trace.activity.unavailable['capture-limit'] : ''}`;
  $('activity-page-control').hidden = view.extent <= config.maxDisplayValues;
  $('activity-page').max = String(Math.max(0, Math.ceil(view.extent / config.maxDisplayValues) - 1));
  canvas.onpointermove = event => {
    const box = canvas.getBoundingClientRect();
    const pixel = Math.floor((event.clientY - box.top) / box.height * side) * side
      + Math.floor((event.clientX - box.left) / box.width * side);
    const index = view.offset + pixel, segment = view.segments.find(s => index >= s.start && index < s.end);
    const value = pixel >= view.values.length ? 'padding' : view.values[pixel] === undefined ? 'missing' : String(view.values[pixel]);
    const block = segment ? ` · ${segment.id}, values ${(index - segment.start) * segment.blockSize}–${Math.min(segment.count, (index - segment.start + 1) * segment.blockSize) - 1}` : '';
    $('activation-cell').hidden = false;
    $('activation-cell').textContent = `Pixel ${index}: ${value}${block} · ${view.representation}`;
  };
  canvas.onpointerleave = () => { $('activation-cell').hidden = true; };
}

function queuePaint() { if (scheduled === null) scheduled = requestAnimationFrame(paint); }

export function clearActivationRecording() {
  revision++; exampleError = '';
  exampleRequest?.abort(); exampleRequest = null; exampleLoading = false;
  $('activity-example').disabled = false;
  stopPlayback(); baseline = null; recording = null; cursor = 0; follow = true; active = false;
  currentPrompt = '';
  document.getElementById('app')?.removeAttribute('data-live-activity');
  queuePaint();
}

export function beginActivationRecording(prompt = '') {
  const mode = activityModes(recording).find(m => m.id === $('activity-mode').value);
  if (!mode || mode.clock === 'snapshot') $('activity-mode').value = config.defaultMode;
  clearActivationRecording();
  if (!isActivationRecordingEnabled()) return;
  currentPrompt = prompt;
  $('activity-layer').value = '-1';
  $('activity-playback').value = activityModes(recording).find(m => m.id === $('activity-mode').value)?.clock === 'boundary' ? 'captures' : 'tokens';
  source = 'Current run'; active = true;
  document.getElementById('app')?.setAttribute('data-live-activity', '');
  const view = $('activation-view'), box = view.getBoundingClientRect();
  if (box.top < 0 || box.top > innerHeight * 0.6) view.scrollIntoView({ block: 'start', behavior: 'instant' });
  queuePaint();
}

/** @type {typeof import('./index.js').acceptActivationFrame} */
export function acceptActivationFrame(event) {
  if (!active || event.type !== 'activation-frame') return;
  const frames = recording?.trace.frames ?? [];
  frames[event.frame.index] = event.frame;
  recording = { schema: 'doppler.activation-recording/v1', demoBuild: SHELL_MANIFEST_DIGEST,
    prompt: currentPrompt,
    display: recording?.display ?? { layout: config.layout, palette: config.palette, scale: config.scale },
    trace: { ...event.trace, frames } };
  if (follow) cursor = frames.length - 1;
  queuePaint();
}

/** @type {typeof import('./index.js').finishActivationRecording} */
export function finishActivationRecording(status, receipt = null, outputText = null) {
  if (!active) return;
  active = false;
  document.getElementById('app')?.removeAttribute('data-live-activity');
  if (recording) {
    if (receipt?.activationTrace) recording.trace = receipt.activationTrace;
    recording.trace.status = status;
    recording.fingerprint = receipt?.fingerprint ?? null;
    recording.inputTokenIds = receipt?.inputTokenIds ?? null;
    recording.generatedTokenIds = receipt?.generatedTokenIds ?? null;
    recording.outputText = receipt?.outputText ?? outputText;
  }
  queuePaint();
}

/** @type {typeof import('./index.js').importActivationRecording} */
export function importActivationRecording(value) {
  const next = validateRecording(value);
  const modes = activityModes(next);
  if (modes.find(m => m.id === $('activity-mode').value)?.reason) $('activity-mode').value = modes.find(m => !m.reason)?.id ?? 'layer-output';
  next.display = { ...next.display, layout: config.layout, palette: config.palette };
  revision++; source = 'Saved recording';
  stopPlayback(); active = false; follow = false; recording = next; cursor = Math.max(0, points().length - 1);
  document.getElementById('app')?.removeAttribute('data-live-activity');
  $('activation-position').max = String(Math.max(...next.trace.frames.map(f => f.tokenStart + f.tokenIds.length - 1)));
  $('activation-position').value = String(points()[cursor]?.position ?? 0);
  $('activation-scale').value = String(next.display.scale);
  queuePaint();
}

/** @type {typeof import('./index.js').isActivationRecordingFile} */
export function isActivationRecordingFile(value) {
  const schema = value && typeof value === 'object' ? /** @type {{ schema?: unknown }} */ (value).schema : undefined;
  return typeof schema === 'string' && schema.startsWith('doppler.activation-recording/');
}

// Opens a standalone recording file; errors stay visible beside the grid.
/** @type {typeof import('./index.js').openActivationRecording} */
export function openActivationRecording(value, byteLength, options = {}) {
  try {
    if (byteLength > config.maxImportBytes) throw new Error('Recording file exceeds the import limit.');
    const next = validateRecording(value);
    onOpenRecording?.(next, options);
    importActivationRecording(next);
  } catch (error) {
    $('activation-status').textContent = String(error);
    throw error;
  }
}

export function getActivationRecording() { return recording; }

/** @type {typeof import('./index.js').initActivationGrid} */
export function initActivationGrid(options = {}) {
  onOpenRecording = options.onOpenRecording ?? null;
  onPlayback = options.onPlayback ?? null;
  unsureWords = options.unsureWords ?? null;
  onUnsureWord = options.onUnsureWord ?? null;
  $('activity-playback').value = config.defaultPlayback;
  for (const [id, key, enabled] of [
    ['activation-record', config.recordPreferenceKey, config.recordByDefault],
  ]) {
    const input = $(/** @type {'activation-record'} */ (id));
    try { const saved = localStorage.getItem(String(key)); input.checked = saved === null ? Boolean(enabled) : saved === 'true'; }
    catch { input.checked = Boolean(enabled); }
    input.addEventListener('change', () => { try { localStorage.setItem(String(key), String(input.checked)); } catch { /* Private storage may be unavailable. */ } });
  }

  $('activation-scale').value = String(config.scale);
  $('activation-scrub').addEventListener('input', () => {
    stopPlayback(); follow = false; openingHint = false; cursor = Number($('activation-scrub').value); $('activation-position').value = String(points()[cursor]?.position ?? 0); queuePaint();
  });
  $('activation-position').addEventListener('input', () => {
    stopPlayback(); follow = false;
    const position = Number($('activation-position').value), timeline = points();
    for (let i = 0; i < timeline.length; i++) if (timeline[i].position === position) cursor = i;
    queuePaint();
  });
  $('activation-scale').addEventListener('change', () => {
    const scale = Number($('activation-scale').value);
    if (!Number.isFinite(scale) || scale <= 0) { $('activation-scale').value = String(recording?.display.scales?.[scaleKey()] ?? recording?.display.scale ?? config.scale); return; }
    if (recording) (recording.display.scales ??= {})[scaleKey()] = scale;
    queuePaint();
  });
  $('activation-live').addEventListener('click', () => { stopPlayback(); follow = true; cursor = points().length - 1; queuePaint(); });
  $('activation-play').addEventListener('click', () => {
    if (!recording) return;
    if (timer !== null) { stopPlayback(); return; }
    follow = false; openingHint = false;
    if (cursor >= points().length - 1) cursor = firstPlayable();
    $('activation-play').textContent = 'Pause';
    const step = () => {
      if (!recording) return;
      paint();
      if (cursor >= points().length - 1) { stopPlayback(); return; }
      const timeline = points();
      const delay = $('activity-playback').value === 'time' ? Math.max(1, timeline[cursor + 1].atMs - timeline[cursor].atMs) : 1000 / config.playbackFps;
      timer = setTimeout(() => { cursor++; $('activation-position').value = String(timeline[cursor].position); step(); }, delay / Number($('activation-speed').value));
    };
    step();
  });
  $('activation-export').addEventListener('click', () => {
    if (!recording) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(encodeRecording(recording))], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'doppler-activation-trace.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  });
  // Keep the token being looked at; a view without values there moves to its nearest drawable token.
  function keepPosition() {
    const position = Number($('activation-position').value), timeline = points();
    cursor = nearestIndex(timeline, position);
    if (timeline.length) $('activation-position').value = String(timeline[cursor].position);
  }
  $('activity-playback').addEventListener('change', () => { stopPlayback(); keepPosition(); queuePaint(); });
  function selectMode() {
    stopPlayback(); openingHint = false;
    const mode = activityModes(recording).find(m => m.id === $('activity-mode').value);
    $('activity-playback').value = mode?.clock === 'boundary' ? 'captures' : 'tokens';
    keepPosition();
    $('activity-page').value = '0'; queuePaint();
  }
  $('activity-go').addEventListener('click', () => {
    const position = Number($('activity-go').dataset.position);
    if (!Number.isFinite(position) || $('activity-go').dataset.position === '') return;
    stopPlayback(); follow = false; openingHint = false;
    $('activation-position').value = String(position);
    keepPosition(); queuePaint();
  });
  $('activity-unsure-prev').addEventListener('click', () => jumpToUnsure(-1));
  $('activity-unsure-next').addEventListener('click', () => jumpToUnsure(1));
  $('activity-mode').addEventListener('change', () => {
    if ($('activity-mode').value === 'model-paths') $('activity-layer').value = '-1';
    selectMode();
  });
  $('activity-paths-button').addEventListener('click', () => { $('activity-mode').value = 'model-paths'; $('activity-layer').value = '-1'; selectMode(); });
  for (const id of ['activity-layer', 'activity-weight', 'activity-detail']) {
    $(/** @type {'activity-mode'} */ (id)).addEventListener('change', () => { stopPlayback(); $('activity-page').value = '0'; queuePaint(); });
  }
  $('activity-page').addEventListener('input', queuePaint);
  $('activity-compare').addEventListener('click', () => $('activity-compare-file').click());
  $('activity-adjacent').addEventListener('click', () => { baseline = null; queuePaint(); });
  $('activity-compare-file').addEventListener('change', async () => {
    try {
      const file = $('activity-compare-file').files?.[0];
      if (!file || !recording) return;
      if (file.size > config.maxImportBytes) throw new Error('Comparison file exceeds the import limit.');
      const next = validateRecording(JSON.parse(await file.text()));
      assertMatchedRuns(recording, next); baseline = next; queuePaint();
    } catch (error) { $('activity-availability').hidden = false; $('activity-availability').textContent = String(error); }
  });
  $('activity-example').addEventListener('click', () => { void loadActivityExample(config.defaultMode); });
  for (const preview of previews) {
    const card = document.createElement('button');
    card.type = 'button'; card.className = 'activity-preview'; card.dataset.mode = preview.mode;
    const img = document.createElement('img');
    img.src = new URL(`../../assets/activity-preview-${preview.mode}.png`, import.meta.url).href;
    img.alt = ''; img.width = 144; img.height = 144;
    const label = document.createElement('span'); label.textContent = preview.label;
    const note = document.createElement('small'); note.textContent = preview.description;
    const cadence = document.createElement('em'); cadence.className = 'activity-cadence';
    const clock = activityModes(null).find(m => m.id === preview.mode)?.clock;
    cadence.textContent = clock === 'snapshot' ? 'Snapshot' : clock === 'boundary' ? 'Captured updates' : 'Token playback';
    card.append(img, label, cadence, note);
    const data = previewData.previews.find(p => p.mode === preview.mode);
    if (data) {
      const key = document.createElement('small'); key.className = 'activity-preview-key';
      const colors = data.range === 'signed' ? config.colors.signed : config.colors.positive;
      key.style.borderImage = `linear-gradient(to right, ${colors.map(c => `rgb(${c.join(',')})`).join(',')}) 1`;
      key.textContent = `${data.range === 'signed' ? '−' + Number(data.scale.toPrecision(2)) : '0'} → ${Number(data.scale.toPrecision(2))}`;
      card.append(key);
    }
    card.addEventListener('click', () => {
      if (!recording) { void loadActivityExample(preview.mode); return; }
      $('activity-mode').value = preview.mode; $('activity-detail').value = 'overview';
      $('activity-layer').value = '-1'; $('activity-weight').value = ''; selectMode();
    });
    $('activity-gallery').append(card);
  }
  $('activation-record').addEventListener('change', queuePaint);
  document.addEventListener('visibilitychange' , () => { if (document.hidden) stopPlayback(); });
  globalThis.matchMedia?.('(prefers-color-scheme: dark)').addEventListener('change', queuePaint);
  queuePaint();
}

/** @type {typeof import('./index.js').loadActivityExample} */
export async function loadActivityExample(mode = config.defaultMode, { opening = false } = {}) {
  if (exampleLoading || active) return;
  const expectedRevision = revision;
  const request = new AbortController(); exampleRequest = request;
  exampleLoading = true; exampleError = '';
  $('activity-source').hidden = false;
  $('activity-source').textContent = 'Opening the recorded Qwen 0.8B example…';
  for (const card of $('activity-gallery').querySelectorAll('button')) card.disabled = true;
  $('activity-example').disabled = true;
  try {
    const metadataResponse = await fetch(new URL('../../assets/activity-example.json', import.meta.url), { signal: request.signal });
    if (!metadataResponse.ok) throw new Error('Example description is unavailable.');
    const metadata = await metadataResponse.json();
    const response = await fetch(new URL('../../assets/activity-example.json.gz', import.meta.url), { signal: request.signal });
    if (!response.ok) throw new Error('Example recording is unavailable.');
    const bytes = await response.arrayBuffer();
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(n => n.toString(16).padStart(2, '0')).join('');
    if (hash !== metadata.sha256 || bytes.byteLength !== metadata.compressedBytes) throw new Error('Example recording integrity check failed.');
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    const text = await new Response(stream).text();
    if (text.length !== metadata.decodedBytes) throw new Error('Example recording length differs.');
    if (revision !== expectedRevision || active) return;
    $('activity-playback').value = mode === 'model-paths' ? 'captures' : 'tokens';
    $('activity-layer').value = '-1';
    $('activity-detail').value = 'overview';
    $('activity-weight').value = '';
    $('activity-page').value = '0';
    openActivationRecording({ ...JSON.parse(text), prompt: metadata.prompt }, text.length, { preserveDraft: opening });
    // The list now holds this recording's views; only then can the requested one be chosen.
    syncViewList(activityModes(recording));
    if ([...$('activity-mode').options].some(option => option.value === mode && !option.disabled)) $('activity-mode').value = mode;
    source = 'Recorded replay';
    cursor = 0; const examplePoints = points();
    // A first visit opens paused on the word the model was least sure of, so the
    // picture is of a moment worth looking at. Play shows the answer from the start.
    const leastSure = opening ? [...(unsureWords?.() ?? [])].sort((a, b) => a.probability - b.probability)
      .map(word => examplePoints.findIndex(point => point.position === word.position)).find(index => index >= 0) : undefined;
    if (leastSure !== undefined) { cursor = leastSure; openingHint = true; }
    else for (let i = 0; i < examplePoints.length; i++) if (examplePoints[i].position === metadata.promptPosition) cursor = i;
    $('activation-position').value = String(examplePoints[cursor]?.position ?? 0);
    paint();
    if (leastSure !== undefined) onUnsureWord?.(examplePoints[cursor].position);
  } catch (error) {
    if (!request.signal.aborted) { exampleError = String(error); queuePaint(); }
  }
  finally {
    if (exampleRequest === request) {
      exampleRequest = null; exampleLoading = false; $('activity-example').disabled = false;
      queuePaint();
    }
  }
}

export function cancelActivityExample() {
  if (exampleRequest) {
    revision++;
    exampleRequest.abort(); exampleRequest = null; exampleLoading = false;
    $('activity-example').disabled = false;
    queuePaint();
  }
  stopPlayback();
}
