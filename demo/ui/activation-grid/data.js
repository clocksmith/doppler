import { decodeRecording } from './recording-codec.js';
import { validateActivity } from './validation.js';
import config from './config.json' with { type: 'json' };

// Import validation is independent of a recording's claimed limits.
/** @type {typeof import('./data.js').validateRecording} */
export function validateRecording(input) {
  input = decodeRecording(input);
  if (!input || typeof input !== 'object') throw new Error('Invalid activation recording.');
  const value = /** @type {import('./data.js').ActivationRecording} */ (input);
  const fail = () => { throw new Error('Invalid or oversized activation recording.'); };
  if (value?.schema !== 'doppler.activation-recording/v1' || typeof value.demoBuild !== 'string' || !value.demoBuild
    || !config.compatibleDisplays.some(d => d.layout === value.display?.layout && d.palette === value.display?.palette)
    || !Number.isFinite(value.display.scale) || value.display.scale <= 0) fail();
  if (value.display.scales !== undefined && (!value.display.scales || typeof value.display.scales !== 'object'
    || Array.isArray(value.display.scales) || Object.keys(value.display.scales).length > 64
    || Object.values(value.display.scales).some(n => !Number.isFinite(n) || n <= 0))) fail();
  const trace = value.trace, geometry = trace?.geometry;
  if (trace?.schema !== 'doppler.activation-trace/v1' || !trace.identity || !trace.capturePolicy
    || ![trace.identity.modelId, trace.identity.runtimeVersion].every(s => typeof s === 'string' && s.length > 0)
    || !(trace.identity.manifestHash === null || typeof trace.identity.manifestHash === 'string')
    || !['complete', 'recording', 'stopped', 'error'].includes(trace.status)
    || !Object.values(trace.capturePolicy).every(n => Number.isSafeInteger(n) && n > 0)
    || !['maxBytes', 'maxFrameBytes', 'maxFrames', 'maxGridValues', 'maxInFlight'].every(k => k in trace.capturePolicy)
    || ![geometry?.layers, geometry?.width].every(n => Number.isSafeInteger(n) && n > 0)
    || geometry.layers * geometry.width > config.maxGridValues
    || !Array.isArray(trace.frames) || trace.frames.length > config.maxFrames) fail();
  let total = 0, tokenMetadata = 0, previousForward = -1, previousTime = -1;
  for (const [index, f] of trace.frames.entries()) {
    if (f?.index !== index || !Number.isSafeInteger(f.forward) || f.forward < 0
      || !['prefill', 'decode'].includes(f.phase) || f.role !== 'layer-output'
      || !Number.isSafeInteger(f.layer) || f.layer < 0 || f.layer >= geometry.layers
      || !Number.isSafeInteger(f.tokenStart) || f.tokenStart < 0 || !Number.isFinite(f.atMs) || f.atMs < 0
      || f.forward < previousForward || f.atMs < previousTime
      || !Array.isArray(f.tokenIds) || !f.tokenIds.length || f.tokenIds.length > config.maxValues
      || f.tokenIds.some(n => !Number.isSafeInteger(n) || n < 0)
      || !Array.isArray(f.shape) || f.shape.length !== 2 || f.shape[0] !== f.tokenIds.length || f.shape[1] !== geometry.width
      || !['f16', 'f32'].includes(f.dtype) || ![null, 'two-f32'].includes(f.precision)
      || !['pending', 'captured', 'failed', 'dropped'].includes(f.status)) fail();
    previousForward = f.forward; previousTime = f.atMs;
    tokenMetadata += f.tokenIds.length;
    if (tokenMetadata > config.maxValues) fail();
    if (f.status === 'captured') {
      if (!Array.isArray(f.values) || f.values.length !== f.tokenIds.length * geometry.width) fail();
      const samples = f.values;
      if (!samples) fail();
      total += samples?.length ?? 0;
      if (total > config.maxValues || samples?.some(n => !(typeof n === 'number' && Number.isFinite(n))
        && !['NaN', 'Infinity', '-Infinity'].includes(String(n)))) fail();
    } else if (f.values != null) fail();
  }
  if (trace.recordedBytes !== total * 8) fail();
  if (trace.activity) validateActivity(trace.activity, trace.geometry);
  return value;
}

// This is an explicitly labeled composite of sequential captures from ONE
// forward pass, never an assertion of simultaneous state or uncaptured zeros.
/** @type {typeof import('./data.js').projectFrame} */
export function projectFrame(trace, index, position) {
  const { layers, width } = trace.geometry;
  /** @type {Array<number | string | undefined>} */
  const values = new Array(layers * width);
  /** @type {Array<import('doppler-gpu/tooling/evidence').ActivationFrame | undefined>} */
  const sources = new Array(layers);
  const selected = trace.frames[index];
  for (let i = index; i >= 0 && trace.frames[i]?.forward === selected.forward; i--) {
    const f = trace.frames[i];
    if (sources[f.layer]) continue;
    sources[f.layer] = f;
    if (f.status !== 'captured' || !f.values || position < f.tokenStart || position >= f.tokenStart + f.tokenIds.length) continue;
    const offset = (position - f.tokenStart) * width;
    for (let d = 0; d < width; d++) values[f.layer * width + d] = f.values[offset + d];
  }
  return { values, sources };
}

/** @type {typeof import('./data.js').heatmapColor} */
export function heatmapColor(value, scale, range = 'signed') {
  if (!Number.isFinite(value)) return config.colors.invalid;
  const colors = range === 'signed' ? config.colors.signed : config.colors.positive;
  const t = Math.min(1, Math.max(0, range === 'signed' ? (value / scale + 1) / 2 : value / scale));
  const position = t * (colors.length - 1), lo = Math.floor(position), hi = Math.min(lo + 1, colors.length - 1);
  return colors[lo].map((v, i) => Math.round(v + (colors[hi][i] - v) * (position - lo)));
}

/** @type {typeof import('./data.js').colorGrid} */
export function colorGrid(values, scale, range = 'signed') {
  const side = Math.ceil(Math.sqrt(values.length));
  const rgba = new Uint8ClampedArray(side * side * 4);
  for (let p = 0; p < side * side; p++) {
    const value = values[p];
    const rgb = p >= values.length ? config.colors.padding : value === undefined
      ? config.colors.missing[(p % side + Math.floor(p / side)) % 2]
      : typeof value === 'number' ? heatmapColor(value, scale, range) : config.colors.invalid;
    rgba.set(rgb, p * 4);
    rgba[p * 4 + 3] = 255;
  }
  return { side, rgba };
}

// Freeze this initial 99th-percentile magnitude per quantity/window for playback.
// Changing token position never renormalizes an already chosen scale.
/** @type {typeof import('./data.js').initialColorScale} */
export function initialColorScale(values, fallback) {
  const magnitudes = values.filter(v => typeof v === 'number' && Number.isFinite(v) && v !== 0).map(v => Math.abs(Number(v))).sort((a, b) => a - b);
  return magnitudes.length ? magnitudes[Math.min(magnitudes.length - 1, Math.floor(magnitudes.length * 0.99))] : fallback;
}
