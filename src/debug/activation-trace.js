import { createActivityTrace } from './activity-trace.js';
import { snapshotTensor, snapshotFromArray, snapshotElementBytes } from './tensor.js';

// Copies are enqueued before the caller can reuse its activation buffer. Mapping
// runs asynchronously, with bounded staging and explicit gaps under backpressure.
/** @type {typeof import('./activation-trace.js').createActivationTrace} */
export function createActivationTrace(limits, geometry, identity, onFrame) {
  /** @type {Array<'maxBytes' | 'maxFrameBytes' | 'maxFrames' | 'maxGridValues' | 'maxInFlight'>} */
  const fields = ['maxBytes', 'maxFrameBytes', 'maxFrames', 'maxGridValues', 'maxInFlight'];
  for (const key of fields) {
    if (!Number.isSafeInteger(limits[key]) || limits[key] < 1) throw new Error(`Activation trace requires ${key}.`);
  }
  if (![geometry.layers, geometry.width].every(n => Number.isSafeInteger(n) && n > 0)
    || geometry.layers * geometry.width > limits.maxGridValues) throw new Error('Activation trace geometry exceeds its declared limit.');
  /** @type {import('./activation-trace.js').ActivationFrame[]} */
  const frames = [];
  /** @type {Set<Promise<void>>} */
  const pending = new Set();
  let bytes = 0, reserved = 0, forward = -1;
  /** @type {string | null} */
  let stoppedReason = null;
  /** @type {unknown} */
  let observerError = null;
  const started = performance.now();
  const activity = limits.activityMaxBytes ? createActivityTrace(/** @type {import('./activity-trace.js').ActivityLimits} */ (limits), started) : undefined;
  /** @param {import('./activation-trace.js').ActivationFrame} frame */
  const publish = frame => {
    frames[frame.index] = Object.freeze(frame);
    try { onFrame?.(frames[frame.index]); } catch (error) { observerError ??= error; }
  };
  return {
    activity,
    capture(buffer, context) {
      if (stoppedReason) return;
      // Submit the producer before the independent staging copy; keep temporary
      // resources owned by the recorder until its normal completion/abort path.
      context.recorder?.submitPartial();
      const { layer, tokenStart, tokenIds, dtype } = context;
      if (layer === 0) forward++;
      if (!Number.isSafeInteger(layer) || layer < 0 || layer >= geometry.layers
        || !Number.isSafeInteger(tokenStart) || tokenStart < 0 || !tokenIds.length) throw new Error('Invalid activation capture boundary.');
      const shape = [tokenIds.length, geometry.width];
      const elements = tokenIds.length * geometry.width;
      const payloadBytes = elements * 8; // retained JS numbers, excluding metadata/engine overhead
      const readBytes = elements * snapshotElementBytes(dtype, context);
      /** @type {import('./activation-trace.js').ActivationFrame} */
      const frame = { index: frames.length, forward, atMs: performance.now() - started,
        phase: context.phase, role: 'layer-output', layer, tokenStart,
        tokenIds: Object.freeze([...tokenIds]), shape: Object.freeze(shape), dtype,
        precision: context.precision ?? null, status: 'pending', values: null, reason: null };
      if (payloadBytes > limits.maxFrameBytes || readBytes > limits.maxFrameBytes
        || bytes + reserved + payloadBytes > limits.maxBytes || frames.length >= limits.maxFrames - 1) {
        stoppedReason = 'Recording limit reached; generation continues.';
        publish({ ...frame, status: 'dropped', reason: stoppedReason });
        return;
      }
      if (pending.size >= limits.maxInFlight) {
        publish({ ...frame, status: 'dropped', reason: 'Readback queue full.' });
        return;
      }
      const cpu = buffer instanceof Float32Array || buffer instanceof Float64Array;
      if (cpu ? buffer.length < elements : buffer.size < readBytes) throw new Error('Activation buffer is smaller than its declared shape.');
      publish(frame);
      reserved += payloadBytes;
      const options = { ...context, includeData: true };
      // snapshotTensor enqueues its staging copy synchronously before mapAsync.
      const read = cpu ? Promise.resolve(snapshotFromArray(buffer, shape, dtype, options))
        : snapshotTensor(buffer, shape, dtype, options);
      const job = read.then(captured => {
        if (!captured.ok || captured.data?.length !== elements) {
          publish({ ...frame, status: 'failed', reason: captured.error ?? 'Incomplete tensor.' });
          return;
        }
        bytes += payloadBytes;
        publish({ ...frame, status: 'captured', values: Object.freeze(captured.data.map(v => Number.isFinite(v) ? v : String(v))) });
      }).catch(error => { publish({ ...frame, status: 'failed', reason: String(error) }); })
        .finally(() => { reserved -= payloadBytes; pending.delete(job); });
      pending.add(job);
    },
    async drain() {
      await Promise.all(pending);
      if (observerError) throw observerError;
    },
    snapshot(status, includeFrames = true) {
      return { schema: 'doppler.activation-trace/v1', identity: { ...identity }, geometry: { ...geometry },
        capturePolicy: { ...limits }, ...(activity ? { activity: activity.snapshot(true) } : {}), status, stoppedReason, recordedBytes: bytes, frames: includeFrames ? [...frames] : [] };
    },
  };
}
