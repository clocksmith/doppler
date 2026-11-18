/** Opt-in diagnostic capture. Never enabled by profiling or ordinary inference. */
/** @type {WeakMap<object, import('./routed-expert-trace.js').RoutedExpertTrace>} */
const traces = new WeakMap();
/** @type {WeakMap<import('./routed-expert-trace.js').RoutedExpertTrace, { records: number, indices: number }>} */
const pending = new WeakMap();

/** @type {typeof import('./routed-expert-trace.js').beginRoutedExpertTrace} */
export function beginRoutedExpertTrace(loader, options) {
  if (!options || typeof options.requestId !== 'string' || !options.requestId.trim()
    || !Number.isSafeInteger(options.maxRecords) || options.maxRecords < 1
    || !Number.isSafeInteger(options.maxIndices) || options.maxIndices < 1) {
    throw new Error('Expert trace requires a request ID and explicit positive record/index limits.');
  }
  if (traces.has(loader)) throw new Error('An expert trace is already active for this loader.');
  traces.set(loader, { schema: 'doppler.routed-expert-trace/v1', requestId: options.requestId,
    maxRecords: options.maxRecords, maxIndices: options.maxIndices,
    scope: 'Observed source expert selections before deduplication/remapping; not proof of completed execution. Diagnostic resident reads add overhead.',
    records: [], capturedIndices: 0, omittedRecords: 0, omittedIndices: 0 });
}

/** @type {typeof import('./routed-expert-trace.js').finishRoutedExpertTrace} */
export function finishRoutedExpertTrace(loader, requestId) {
  const trace = traces.get(loader);
  if (!trace || trace.requestId !== requestId) throw new Error('Expert trace request ID does not match the active capture.');
  if (pending.get(trace)?.records) throw new Error('Expert trace requires all in-flight reads to settle before finishing.');
  traces.delete(loader);
  return structuredClone(trace);
}

/** @type {typeof import('./routed-expert-trace.js').cancelRoutedExpertTrace} */
export function cancelRoutedExpertTrace(loader) { traces.delete(loader); }

/** @type {typeof import('./routed-expert-trace.js').isRoutedExpertTraceActive} */
export function isRoutedExpertTraceActive(loader) { return traces.has(loader); }

/** Reuses mandatory paging reads; resident reads happen only for an admitted capture.
 * @type {typeof import('./routed-expert-trace.js').readRoutedExpertSelection} */
export async function readRoutedExpertSelection(loader, metadata, read, required) {
  const trace = traces.get(loader);
  const length = metadata.numTokens * metadata.topK;
  if (trace && (!Number.isSafeInteger(length) || length < 1
    || !Number.isSafeInteger(metadata.layer) || metadata.layer < 0
    || !Number.isSafeInteger(metadata.tokenOffset) || metadata.tokenOffset < 0
    || !Number.isSafeInteger(metadata.numTokens) || metadata.numTokens < 1
    || !Number.isSafeInteger(metadata.topK) || metadata.topK < 1)) {
    throw new Error('Expert trace requires valid layer and token geometry.');
  }
  const reserved = trace ? (pending.get(trace) ?? { records: 0, indices: 0 }) : null;
  const admitted = trace && reserved && trace.records.length + reserved.records < trace.maxRecords
    && length <= trace.maxIndices - trace.capturedIndices - reserved.indices;
  if (trace && !admitted) {
    trace.omittedRecords++;
    trace.omittedIndices += length;
  }
  if (!required && !admitted) return null;
  if (admitted) {
    reserved.records++;
    reserved.indices += length;
    pending.set(trace, reserved);
  }
  try {
    const data = await read();
    // A previous request may finish or cancel while a read is in flight.
    if (admitted && traces.get(loader) === trace) {
      const indices = new Uint32Array(data);
      if (indices.length !== length) throw new Error('Expert trace readback does not match its declared token geometry.');
      trace.records.push({ ...metadata, indices: Array.from(indices) });
      trace.capturedIndices += indices.length;
    }
    return data;
  } catch (error) {
    if (admitted) { trace.omittedRecords++; trace.omittedIndices += length; }
    throw error;
  } finally {
    if (admitted) { reserved.records--; reserved.indices -= length; }
  }
}
