import config from './config.json' with { type: 'json' };

/** @type {typeof import('./validation.js').validateActivity} */
export function validateActivity(value, geometry) {
  const fail = () => { throw new Error('Invalid or oversized model activity recording.'); };
  const positive = (/** @type {unknown} */ n) => typeof n === 'number' && Number.isSafeInteger(n) && n > 0;
  const number = (/** @type {unknown} */ n) => typeof n === 'number' && Number.isFinite(n);
  const scalar = (/** @type {unknown} */ n) => number(n) || ['NaN', 'Infinity', '-Infinity'].includes(String(n));
  if (!value || value.schema !== 'doppler.model-activity/v1' || !value.model || !value.unavailable
    || !Array.isArray(value.frames) || value.frames.length > config.maxFrames
    || !Array.isArray(value.weights) || value.weights.length > config.maxFrames) fail();
  if (!positive(value.model.intermediateSize) || Number(value.model.intermediateSize) * geometry.layers > config.maxGridValues) fail();
  let bytes = 0, previousTime = -1;
  for (const [index, f] of value.frames.entries()) {
    if (f.index !== index || !Number.isSafeInteger(f.forward) || f.forward < 0 || !number(f.atMs) || f.atMs < previousTime
      || !['prefill', 'decode'].includes(f.phase) || typeof f.role !== 'string' || f.role.length > 80
      || !Number.isInteger(f.layer) || f.layer < -1 || f.layer >= geometry.layers
      || !Number.isSafeInteger(f.tokenStart) || f.tokenStart < 0
      || !Array.isArray(f.tokenIds) || !f.tokenIds.length || f.tokenIds.length > 65536 || f.tokenIds.some(n => !Number.isSafeInteger(n) || n < 0)
      || !Array.isArray(f.fullShape) || !f.fullShape.length || f.fullShape.length > 4 || !f.fullShape.every(positive)
      || f.fullShape.reduce((a, b) => a * b, 1) > 1e12
      || !Array.isArray(f.shape) || f.shape.length !== 1 || !positive(f.shape[0]) || f.shape[0] > config.maxActivityValues
      || !['f32', 'f16', 'u32'].includes(f.dtype) || f.offset !== 0
      || !['raw', 'reconstructed', 'aggregated'].includes(f.representation)
      || !f.metadata || typeof f.metadata !== 'object'
      || !['captured', 'failed', 'dropped'].includes(f.status)) fail();
    previousTime = f.atMs;
    if (f.shape[0] > f.fullShape.reduce((a, b) => a * b, 1)) fail();
    if (f.role === 'weight-access' && (!Array.isArray(f.metadata.weightIds) || f.metadata.weightIds.some(id => typeof id !== 'string' || id.length > 512))) fail();
    if (f.role === 'attention-query') {
      for (const key of ['heads', 'kvHeads', 'headDim', 'kvLength']) if (!positive(f.metadata[key]) || Number(f.metadata[key]) > 65536) fail();
      if (!number(f.metadata.scale) || !number(f.metadata.softcap) || (f.metadata.slidingWindow !== null && !number(f.metadata.slidingWindow))
        || typeof f.metadata.causal !== 'boolean' || Number(f.metadata.heads) % Number(f.metadata.kvHeads)
        || f.fullShape[1] !== Number(f.metadata.heads) * Number(f.metadata.headDim)) fail();
    }
    if (f.role.startsWith('expert-') && (!positive(f.metadata.numExperts) || Number(f.metadata.numExperts) > 65536)) fail();
    if (f.status === 'captured') {
      if (!Array.isArray(f.values) || f.values.length !== f.shape[0] || !f.values.every(scalar)) fail();
      bytes += (f.values?.length ?? 0) * 8;
      if (f.summary) {
        if (!positive(f.summary.blockSize) || !positive(f.summary.observedValues) || !Array.isArray(f.summary.values)
          || f.summary.values.length !== Math.ceil(f.summary.observedValues / f.summary.blockSize)
          || !f.summary.values.every(scalar)) fail();
        bytes += f.summary.values.length * 8;
      }
      if (bytes > config.maxActivityValues * 8) fail();
    } else if (f.values != null) fail();
  }
  if (bytes !== value.recordedBytes) fail();
  let stored = 0;
  const identities = new Set();
  for (const w of value.weights) {
    if (typeof w.id !== 'string' || !w.id || identities.has(w.id) || w.id.length > 512
      || !(w.layer === null || Number.isInteger(w.layer) && w.layer >= 0 && w.layer < geometry.layers)
      || !Array.isArray(w.shape) || w.shape.length > 4 || !w.shape.every(positive)
      || w.shape.reduce((a, b) => a * b, 1) > 1e12
      || !positive(w.blockSize) || !positive(w.rowGroupSize) || w.rawOffset !== 0
      || typeof w.resident !== 'boolean' || !number(w.byteLength) || w.byteLength < 0) fail();
    identities.add(w.id);
    for (const list of [w.rms, w.raw, w.columnEnergy, w.error ?? []]) {
      if (!Array.isArray(list) || !list.every(number)) fail();
      stored += list.length;
      if (stored > config.maxWeightValues) fail();
    }
    if (w.error?.length && (!value.weights.some(r => r.id === w.referenceId && r.dtype === w.referenceDtype && r.raw.length === w.error?.length) || w.raw.length !== w.error.length || value.model.useTiedEmbeddings !== true)) fail();
    const count = w.shape.length ? w.shape.reduce((a, b) => a * b, 1) : 0;
    if (w.rms.length && (w.rms.length !== Math.ceil(count / w.blockSize) || w.rms.some(n => n < 0))) fail();
    if (w.raw.length > count || (w.error?.length ?? 0) > count || w.columnEnergy.some(n => n < 0)) fail();
    if (w.columnEnergy.length && (w.shape.length !== 2 || w.columnEnergy.length !== Math.ceil(w.shape[0] / w.rowGroupSize) * w.shape[1])) fail();
  }
}
