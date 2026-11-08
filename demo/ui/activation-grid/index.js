import config from './config.json' with { type: 'json' };
import previews from './previews.json' with { type: 'json' };
import { SHELL_MANIFEST_DIGEST } from '../../generated-shell-manifest.js';
import { validateRecording, colorGrid, initialColorScale } from './data.js';
import { encodeRecording } from './recording-codec.js';
import { activityView, activityModes, playbackPoints, assertMatchedRuns } from './views.js';

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
let source = 'Saved recording';

export function isActivationRecordingEnabled() { return $('activation-record')?.checked === true; }

function stopPlayback() {
  if (timer !== null) clearTimeout(timer);
  timer = null;
  if ($('activation-play')) $('activation-play').textContent = 'Play';
}

function scaleKey() { return `${$('activity-mode').value}:${$('activity-detail').value}`; }

function points() { return recording ? playbackPoints(recording, $('activity-playback').value) : []; }

function paint() {
  scheduled = null;
  const trace = recording?.trace, timeline = points();
  $('activity-previews').hidden = timeline.length > 0 || active;
  $('activity-selectors').hidden = !timeline.length;
  $('activation-caption').hidden = !timeline.length && !active;
  $('activity-source').hidden = !timeline.length && !exampleLoading;
  if (!exampleLoading && recording) $('activity-source').textContent = `${source} · ${trace?.identity.logicalModelId ?? trace?.identity.modelId}`;
  const select = $('activity-mode'), availability = activityModes(recording);
  if (!select.options.length) for (const mode of availability) select.add(new Option(mode.label, mode.id));
  for (const option of select.options) {
    const mode = availability.find(m => m.id === option.value);
    option.disabled = Boolean(mode?.reason);
    option.title = mode?.reason ?? mode?.label ?? option.value;
    option.textContent = `${mode?.label}${mode?.reason ? ' · unavailable' : ''}`;
  }
  const chosen = availability.find(m => m.id === select.value);
  $('activity-availability').textContent = timeline.length ? chosen?.reason ?? '' : '';
  $('activity-availability').hidden = !timeline.length || !chosen?.reason;
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
    $('activation-canvas').width = 1; $('activation-canvas').height = 1;
    $('activation-layout').textContent = ''; $('activation-status').textContent = '';
    $('activation-caption').textContent = active ? 'Recording… inspecting fixed weights, then model activity.' : '';
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
  const view = activityView(recording, { mode: select.value, atMs: point.atMs, position,
    layer: Number(layerSelect.value), weightId: weightSelect.value, detail: $('activity-detail').value,
    page: Number($('activity-page').value), baseline });
  const scales = recording.display.scales ??= {};
  const key = scaleKey();
  if (view.values.some(v => typeof v === 'number' && v !== 0) && scales[key] === undefined) {
    scales[key] = initialColorScale(view.values, recording.display.scale);
  }
  const scale = scales[key] ?? recording.display.scale;
  $('activation-scale').value = String(scale);
  const { side, rgba } = colorGrid(view.values, scale);
  const canvas = $('activation-canvas'); canvas.width = Math.max(1, side); canvas.height = Math.max(1, side);
  canvas.setAttribute('aria-label', `${view.quantity}: three scalar values per RGB pixel, ${view.representation}`);
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Activity grids require a canvas 2D context.');
  if (side) { const image = context.createImageData(side, side); image.data.set(rgba); context.putImageData(image, 0, 0); }
  const present = view.values.filter(v => v !== undefined).length;
  if (!present && !chosen?.reason) {
    $('activity-availability').hidden = false;
    $('activity-availability').textContent = 'No values captured at this position and layer. Select an earlier prompt position or another layer; fused-away stages and uncaptured windows are unavailable.';
  }
  const playMeaning = $('activity-playback').value === 'tokens'
    ? 'Play selects token positions; prefill positions were computed together.'
    : $('activity-playback').value === 'time' ? 'Play follows recorded capture timestamps, including diagnostic overhead.'
      : 'Play advances operator captures at 12 captures/second at 1×.';
  $('activation-caption').textContent = `${view.quantity} · ${view.representation} · ${point.phase} · token position ${position} · ${side} × ${side} pixels · color ±${Number(scale.toPrecision(3))}`;
  $('activation-layout').textContent = `${present.toLocaleString()}/${view.values.length.toLocaleString()} displayed scalars present; ${view.valuesPerScalar} underlying values per scalar (up to ${3 * view.valuesPerScalar} per RGB pixel). Full-detail extent ${view.underlyingValues.toLocaleString()} scalars (${Math.ceil(Math.sqrt(view.underlyingValues / 3)).toLocaleString()}² RGB pixels); current overview/window extent ${view.extent.toLocaleString()} scalars; tile starts at ${view.offset.toLocaleString()}. ${view.description} ${playMeaning} Fixed signed color scale ±${scale}; gray is zero, dark is missing, magenta is non-finite.`;
  const gaps = [...trace.frames, ...(trace.activity?.frames ?? [])].filter(f => ['failed', 'dropped'].includes(f.status)).length;
  $('activation-status').textContent = `${trace.identity.logicalModelId ?? trace.identity.modelId} · ${trace.identity.adapters?.length ? JSON.stringify(trace.identity.adapters) : 'no active adapters recorded'} · manifest ${trace.identity.manifestHash ?? 'unavailable'} · ${trace.status} · ${gaps} missing captures · ${((trace.recordedBytes + (trace.activity?.recordedBytes ?? 0)) / 1048576).toFixed(1)} MiB of retained numerical payload, plus weights and metadata${trace.stoppedReason ? ' · ' + trace.stoppedReason : ''}${trace.activity?.unavailable['capture-limit'] ? ' · ' + trace.activity.unavailable['capture-limit'] : ''}`;
  $('activity-page-control').hidden = view.extent <= config.maxDisplayValues;
  $('activity-page').max = String(Math.max(0, Math.ceil(view.extent / config.maxDisplayValues) - 1));
  canvas.onpointermove = event => {
    const box = canvas.getBoundingClientRect();
    const pixel = Math.floor((event.clientY - box.top) / box.height * side) * side
      + Math.floor((event.clientX - box.left) / box.width * side);
    const start = pixel * 3;
    $('activation-cell').textContent = `Scalars ${view.offset + start}–${view.offset + start + 2}: ${view.values.slice(start, start + 3).map(v => v === undefined ? 'missing' : String(v)).join(', ')} · ${view.representation}`;
  };
}

function queuePaint() { if (scheduled === null) scheduled = requestAnimationFrame(paint); }

export function beginActivationRecording() {
  if (!isActivationRecordingEnabled()) return;
  revision++; source = 'Current run';
  stopPlayback(); baseline = null; recording = null; cursor = 0; follow = true; active = true;
  queuePaint();
}

/** @type {typeof import('./index.js').acceptActivationFrame} */
export function acceptActivationFrame(event) {
  if (!active || event.type !== 'activation-frame') return;
  const frames = recording?.trace.frames ?? [];
  frames[event.frame.index] = event.frame;
  recording = { schema: 'doppler.activation-recording/v1', demoBuild: SHELL_MANIFEST_DIGEST,
    display: recording?.display ?? { layout: config.layout, palette: config.palette, scale: config.scale },
    trace: { ...event.trace, frames } };
  if (follow) cursor = frames.length - 1;
  queuePaint();
}

/** @type {typeof import('./index.js').finishActivationRecording} */
export function finishActivationRecording(status, receipt = null) {
  if (!active) return;
  active = false;
  if (recording) {
    if (receipt?.activationTrace) recording.trace = receipt.activationTrace;
    recording.trace.status = status;
    recording.fingerprint = receipt?.fingerprint ?? null;
    recording.inputTokenIds = receipt?.inputTokenIds ?? null;
    recording.generatedTokenIds = receipt?.generatedTokenIds ?? null;
    recording.outputText = receipt?.outputText ?? null;
  }
  queuePaint();
}

/** @type {typeof import('./index.js').importActivationRecording} */
export function importActivationRecording(value) {
  const next = validateRecording(value);
  revision++; source = 'Saved recording';
  stopPlayback(); active = false; follow = false; recording = next; cursor = Math.max(0, points().length - 1);
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
export function openActivationRecording(value, byteLength) {
  try {
    if (byteLength > config.maxImportBytes) throw new Error('Recording file exceeds the import limit.');
    importActivationRecording(value);
  } catch (error) {
    $('activation-status').textContent = String(error);
    throw error;
  }
}

export function getActivationRecording() { return recording; }

export function initActivationGrid() {
  $('activation-scale').value = String(config.scale);
  $('activation-scrub').addEventListener('input', () => {
    stopPlayback(); follow = false; cursor = Number($('activation-scrub').value); $('activation-position').value = String(points()[cursor]?.position ?? 0); queuePaint();
  });
  $('activation-position').addEventListener('input', () => { follow = false; queuePaint(); });
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
    follow = false;
    if (cursor >= points().length - 1) cursor = 0;
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
  $('activity-playback').addEventListener('change', () => {
    stopPlayback();
    const position = Number($('activation-position').value), timeline = points();
    cursor = 0;
    for (let i = 0; i < timeline.length; i++) if (timeline[i].position === position) cursor = i;
    queuePaint();
  });
  for (const id of ['activity-mode', 'activity-layer', 'activity-weight', 'activity-detail']) {
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
  /** @param {string} mode */
  async function loadExample(mode) {
    if (exampleLoading || active) return;
    const expectedRevision = revision;
    exampleLoading = true;
    $('activity-source').hidden = false;
    $('activity-source').textContent = 'Opening the recorded Qwen 0.8B example…';
    for (const card of $('activity-gallery').querySelectorAll('button')) card.disabled = true;
    $('activity-example').disabled = true;
    try {
      const metadataResponse = await fetch(new URL('../../assets/activity-example.json', import.meta.url));
      if (!metadataResponse.ok) throw new Error('Example description is unavailable.');
      const metadata = await metadataResponse.json();
      const response = await fetch(new URL('../../assets/activity-example.json.gz', import.meta.url));
      if (!response.ok) throw new Error('Example recording is unavailable.');
      const bytes = await response.arrayBuffer();
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(n => n.toString(16).padStart(2, '0')).join('');
      if (hash !== metadata.sha256 || bytes.byteLength !== metadata.compressedBytes) throw new Error('Example recording integrity check failed.');
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      const text = await new Response(stream).text();
      if (text.length !== metadata.decodedBytes) throw new Error('Example recording length differs.');
      if (revision !== expectedRevision || active) return;
      $('activity-playback').value = 'tokens';
      $('activity-layer').value = '-1';
      $('activity-detail').value = 'overview';
      $('activity-weight').value = '';
      $('activity-page').value = '0';
      $('activity-mode').value = mode;
      openActivationRecording(JSON.parse(text), text.length);
      source = 'Recorded example';
      cursor = points().findIndex(p => p.position === metadata.promptPosition);
      $('activation-position').value = String(metadata.promptPosition);
      paint();
      $('activity-mode').focus({ preventScroll: true });
      $('activation-view').scrollIntoView({ block: 'start' });
    } catch (error) { $('activity-availability').hidden = false; $('activity-availability').textContent = String(error); }
    finally {
      exampleLoading = false; $('activity-example').disabled = false;
      for (const card of $('activity-gallery').querySelectorAll('button')) card.disabled = false;
      if (recording) $('activity-source').textContent = `${source} · ${recording.trace.identity.logicalModelId ?? recording.trace.identity.modelId}`;
      else $('activity-source').hidden = true;
    }
  }
  $('activity-example').addEventListener('click', () => { void loadExample('layer-output'); });
  for (const preview of previews) {
    const card = document.createElement('button');
    card.type = 'button'; card.className = 'activity-preview'; card.dataset.mode = preview.mode;
    const img = document.createElement('img');
    img.src = new URL(`../../assets/activity-preview-${preview.mode}.png`, import.meta.url).href;
    img.alt = ''; img.width = 144; img.height = 144;
    const label = document.createElement('span'); label.textContent = preview.label;
    const note = document.createElement('small'); note.textContent = preview.description;
    card.append(img, label, note);
    card.addEventListener('click', () => { void loadExample(preview.mode); });
    $('activity-gallery').append(card);
  }
  document.addEventListener('visibilitychange' , () => { if (document.hidden) stopPlayback(); });
  queuePaint();
}
