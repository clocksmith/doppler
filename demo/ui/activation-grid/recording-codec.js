import config from './config.json' with { type: 'json' };

/** @param {readonly (number | string)[]} values @param {boolean} wide */
function pack(values, wide) {
  const stride = wide ? 8 : 4, bytes = new Uint8Array(values.length * stride), view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) {
    if (wide) view.setFloat64(i * stride, Number(values[i]), true);
    else view.setFloat32(i * stride, Number(values[i]), true);
  }
  let binary = '';
  for (let start = 0; start < bytes.length; start += 32768) binary += String.fromCharCode(...bytes.subarray(start, start + 32768));
  return { encoding: 'base64-le', dtype: wide ? 'f64' : 'f32', length: values.length, data: btoa(binary) };
}

/** @type {typeof import('./recording-codec.js').encodeRecording} */
export function encodeRecording(recording) {
  const trace = recording.trace;
  return { ...recording, schema: 'doppler.activation-recording/v2', trace: { ...trace,
    frames: trace.frames.map(f => ({ ...f, values: f.values ? pack(f.values, f.precision === 'two-f32') : null })),
    ...(trace.activity ? { activity: { ...trace.activity,
      frames: trace.activity.frames.map(f => ({ ...f, values: f.values ? pack(f.values, f.metadata.precision === 'two-f32') : null, ...(f.summary ? { summary: { ...f.summary, values: pack(f.summary.values, true) } } : {}) })),
      weights: trace.activity.weights.map(w => ({ ...w, rms: pack(w.rms, true), raw: pack(w.raw, false),
        columnEnergy: pack(w.columnEnergy, true), ...(w.error ? { error: pack(w.error, true) } : {}) })),
    } } : {}),
  } };
}

/** @type {typeof import('./recording-codec.js').decodeRecording} */
export function decodeRecording(input) {
  if (!input || typeof input !== 'object' || !('schema' in input) || input.schema !== 'doppler.activation-recording/v2') return input;
  // The wire shape differs only in numeric payloads; validation follows decoding.
  const file = /** @type {import('./recording-codec.js').PackedRecording} */ (input);
  let decodedValues = 0;
  const unpack = (/** @type {{encoding: string; dtype: string; length: number; data: string}} */ payload) => {
    const length = Number(payload?.length), stride = payload?.dtype === 'f64' ? 8 : 4;
    if (payload?.encoding !== 'base64-le' || !['f32', 'f64'].includes(String(payload.dtype))
      || !Number.isSafeInteger(length) || length < 0 || length > config.maxValues
      || typeof payload.data !== 'string' || payload.data.length !== 4 * Math.ceil(length * stride / 3)) throw new Error('Invalid packed numerical payload.');
    decodedValues += length;
    if (decodedValues > config.maxValues + config.maxActivityValues + config.maxWeightValues) throw new Error('Packed recording exceeds the import budget.');
    const binary = atob(payload.data);
    if (binary.length !== length * stride) throw new Error('Truncated packed numerical payload.');
    const bytes = Uint8Array.from(binary, c => c.charCodeAt(0)), view = new DataView(bytes.buffer);
    return Array.from({ length }, (_, i) => {
      const n = stride === 8 ? view.getFloat64(i * stride, true) : view.getFloat32(i * stride, true);
      return Number.isFinite(n) ? n : String(n);
    });
  };
  const trace = file.trace, activity = trace?.activity;
  for (const frames of [trace?.frames, activity?.frames ?? [], activity?.weights ?? []]) {
    if (!Array.isArray(frames) || frames.length > config.maxFrames) throw new Error('Invalid packed recording structure.');
  }
  return { ...file, schema: 'doppler.activation-recording/v1', trace: { ...trace,
    frames: trace.frames.map((f) => ({ ...f, values: f.values == null ? null : unpack(f.values) })),
    ...(activity ? { activity: { ...activity,
      frames: activity.frames.map((f) => ({ ...f, values: f.values == null ? null : unpack(f.values), ...(f.summary ? { summary: { ...f.summary, values: unpack(f.summary.values) } } : {}) })),
      weights: activity.weights.map((w) => ({ ...w, rms: unpack(w.rms), raw: unpack(w.raw),
        columnEnergy: unpack(w.columnEnergy), ...(w.error ? { error: unpack(w.error) } : {}) })),
    } } : {}),
  } };
}
