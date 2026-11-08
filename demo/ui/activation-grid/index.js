import config from './config.json' with { type: 'json' };
import { SHELL_MANIFEST_DIGEST } from '../../generated-shell-manifest.js';
import { validateRecording, projectFrame, colorGrid } from './data.js';

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

export function isActivationRecordingEnabled() { return $('activation-record')?.checked === true; }

function stopPlayback() {
  if (timer !== null) clearTimeout(timer);
  timer = null;
  if ($('activation-play')) $('activation-play').textContent = 'Play';
}

function paint() {
  scheduled = null;
  const trace = recording?.trace, frames = trace?.frames ?? [];
  $('activation-export').disabled = !frames.length;
  $('activation-play').disabled = !frames.length;
  $('activation-scrub').disabled = !frames.length;
  if (!frames.length) {
    $('activation-canvas').width = 1;
    $('activation-canvas').height = 1;
    $('activation-layout').textContent = '';
    $('activation-status').textContent = '';
    $('activation-caption').textContent = active ? 'Waiting for the first layer capture…' : 'Record a run or import a recording.';
    return;
  }
  cursor = Math.min(cursor, frames.length - 1);
  const frame = frames[cursor], scrub = $('activation-scrub'), token = $('activation-position');
  scrub.max = String(frames.length - 1); scrub.value = String(cursor);
  token.min = String(frame.tokenStart); token.max = String(frame.tokenStart + frame.tokenIds.length - 1);
  if (follow || Number(token.value) < Number(token.min) || Number(token.value) > Number(token.max)) token.value = token.max;
  if (!recording || !trace) return;
  const position = Number(token.value), { values, sources } = projectFrame(trace, cursor, position);
  const { side, rgba } = colorGrid(values, recording.display.scale);
  const canvas = $('activation-canvas');
  canvas.width = side; canvas.height = side;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Activation grid requires a canvas 2D context.');
  const image = context.createImageData(side, side); image.data.set(rgba);
  context.putImageData(image, 0, 0);
  const captured = sources.filter(f => f?.status === 'captured').length;
  $('activation-caption').textContent = `${frame.phase === 'prefill' ? 'Prefill' : 'Decode'} · forward ${frame.forward + 1} · input position ${position} · token ID ${frame.tokenIds[position - frame.tokenStart]} · layer ${frame.layer + 1}/${trace.geometry.layers} · capture ${cursor + 1}/${frames.length} · ${frame.status}`;
  $('activation-layout').textContent = `${side} × ${side} RGB cells · ${trace.geometry.layers} layers × ${trace.geometry.width} values · ${captured} captured layers in this composite. Three consecutive values per pixel; no subsampling. Zero is gray, missing is dark, nonfinite is magenta. Scale ±${recording.display.scale}; colors saturate smoothly.`;
  const gaps = frames.filter(f => ['failed', 'dropped'].includes(f.status)).length;
  $('activation-status').textContent = `${trace.identity.modelId} · runtime ${trace.identity.runtimeVersion} · ${trace.status} · ${(trace.recordedBytes / 1048576).toFixed(1)} MiB numerical payload · ${gaps} gaps${trace.stoppedReason ? ' · ' + trace.stoppedReason : ''}`;
  canvas.onpointermove = event => {
    const box = canvas.getBoundingClientRect();
    const pixel = Math.floor((event.clientY - box.top) / box.height * side) * side
      + Math.floor((event.clientX - box.left) / box.width * side);
    const dimension = pixel * 3, layer = Math.floor(dimension / trace.geometry.width), source = sources[layer];
    $('activation-cell').textContent = layer >= trace.geometry.layers ? 'Padding'
      : `Layer ${layer + 1}, dimension ${dimension % trace.geometry.width}: ${values.slice(dimension, dimension + 3).map(v => v === undefined ? 'missing' : String(v)).join(', ')} · ${source ? 'capture ' + (source.index + 1) + ', ' + source.atMs.toFixed(2) + ' ms after observation began, ' + source.status : 'not captured'}`;
  };
}

function queuePaint() { if (scheduled === null) scheduled = requestAnimationFrame(paint); }

export function beginActivationRecording() {
  if (!isActivationRecordingEnabled()) return;
  stopPlayback(); recording = null; cursor = 0; follow = true; active = true;
  $('activation-view').open = true;
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
  stopPlayback(); active = false; follow = false; recording = next; cursor = Math.max(0, next.trace.frames.length - 1);
  $('activation-scale').value = String(next.display.scale);
  $('activation-view').open = true; queuePaint();
}

export function getActivationRecording() { return recording; }

export function initActivationGrid() {
  $('activation-scale').value = String(config.scale);
  $('activation-record').addEventListener('change', () => {
    $('activation-notice').textContent = isActivationRecordingEnabled()
      ? 'Next run records layer outputs with asynchronous GPU readbacks. Diagnostic timings; recording takes precedence over X-Ray timestamps.'
      : 'Recording is off. Playback of captured data needs no loaded model.';
  });
  $('activation-scrub').addEventListener('input', () => {
    stopPlayback(); follow = false; cursor = Number($('activation-scrub').value); queuePaint();
  });
  $('activation-position').addEventListener('input', () => { follow = false; queuePaint(); });
  $('activation-scale').addEventListener('change', () => {
    const scale = Number($('activation-scale').value);
    if (!Number.isFinite(scale) || scale <= 0) { $('activation-scale').value = String(recording?.display.scale ?? config.scale); return; }
    if (recording) recording.display.scale = scale;
    queuePaint();
  });
  $('activation-live').addEventListener('click', () => { stopPlayback(); follow = true; cursor = (recording?.trace.frames.length ?? 1) - 1; queuePaint(); });
  $('activation-play').addEventListener('click', () => {
    if (!recording) return;
    if (timer !== null) { stopPlayback(); return; }
    follow = false;
    if (cursor >= recording.trace.frames.length - 1) cursor = 0;
    $('activation-play').textContent = 'Pause';
    const step = () => {
      if (!recording) return;
      paint();
      if (cursor >= recording.trace.frames.length - 1) { stopPlayback(); return; }
      timer = setTimeout(() => { cursor++; step(); }, 1000 / (config.playbackFps * Number($('activation-speed').value)));
    };
    step();
  });
  $('activation-export').addEventListener('click', () => {
    if (!recording) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(recording)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'doppler-activation-trace.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  });
  $('activation-import').addEventListener('change', async () => {
    const file = $('activation-import').files?.[0]; if (!file) return;
    try {
      if (file.size > config.maxImportBytes) throw new Error('Recording file exceeds the import limit.');
      importActivationRecording(JSON.parse(await file.text()));
    } catch (error) { $('activation-status').textContent = String(error); }
    finally { $('activation-import').value = ''; }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopPlayback(); });
  queuePaint();
}
