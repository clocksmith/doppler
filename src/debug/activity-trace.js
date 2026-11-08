import { decodeSnapshotData, snapshotElementBytes } from './tensor.js';
import stages from '../config/inspection/activity-stages.json' with { type: 'json' };

/** @type {typeof import('./activity-trace.js').createActivityTrace} */
export function createActivityTrace(limits, started) {
  /** @type {Array<'activityMaxBytes' | 'activityMaxFrameValues' | 'activityMaxStateValues' | 'activityTokenWindow' | 'activitySummaryBlock'>} */
  const required = ['activityMaxBytes', 'activityMaxFrameValues', 'activityMaxStateValues', 'activityTokenWindow', 'activitySummaryBlock'];
  for (const name of required) {
    if (!Number.isSafeInteger(limits[name]) || limits[name] < 1) throw new Error(`Activity capture requires ${name}.`);
  }
  /** @type {import('./activity-trace.js').ActivityFrame[]} */
  const frames = [];
  /** @type {import('./activity-trace.js').ActivityWeight[]} */
  let weights = [];
  /** @type {Record<string, string>} */
  const unavailable = {};
  /** @type {Record<string, unknown>} */
  let model = {};
  /** @type {((buffer: GPUBuffer, offset: number, size: number) => Promise<ArrayBuffer>) | null} */
  let reader = null;
  let bytes = 0, forward = -1, tokenStart = 0;
  /** @type {'prefill' | 'decode'} */
  let phase = 'prefill';
  /** @type {number[]} */
  let tokenIds = [];
  /** @type {import('./activity-trace.js').ActivityTrace} */
  const api = {
    begin(nextPhase, start, ids) {
      if (nextPhase === phase && start === tokenStart && tokenIds.length && ids.join(',') === tokenIds.join(',')) return;
      phase = nextPhase; tokenStart = start; tokenIds = [...ids]; forward++;
    },
    setReader(value) { reader = value; },
    configure(config) { model = structuredClone(config); },
    async capture(role, buffer, options) {
      if (role === 'ffn-down-input' && !weights.some(w => w.layer === options.layer && w.columnEnergy.length)) return;
      if (!tokenIds.length) return;
      if (frames.length >= limits.maxFrames) { unavailable['capture-limit'] = 'Activity frame limit reached; later boundaries are unrecorded.'; return; }
      const originalShape = [...options.shape];
      let ids = [...(options.tokenIds ?? tokenIds)], start = options.tokenStart ?? tokenStart;
      let offset = options.offset ?? 0;
      const fullShape = [...originalShape];
      if (role.startsWith('ffn-') && fullShape.length === 2 && fullShape[0] === ids.length && ids.length > limits.activityTokenWindow) {
        const skipped = ids.length - limits.activityTokenWindow;
        offset += skipped * fullShape[1]; start += skipped; ids = ids.slice(skipped); fullShape[0] = ids.length;
      }
      if (!fullShape.length || fullShape.some(n => !Number.isSafeInteger(n) || n <= 0)) throw new Error('Activity capture requires explicit positive dimensions.');
      const total = fullShape.reduce((a, b) => a * b, 1);
      const cap = options.state ? limits.activityMaxStateValues : limits.activityMaxFrameValues;
      const count = Math.min(total, cap);
      const elementBytes = snapshotElementBytes(options.dtype, options);
      const readCount = options.state ? Math.min(total, Math.floor(limits.maxFrameBytes / elementBytes)) : count;
      const summaryCount = options.state ? Math.ceil(readCount / limits.activitySummaryBlock) : 0;
      const payloadBytes = (count + summaryCount) * 8;
      /** @type {import('./activity-trace.js').ActivityFrame} */
      const frame = { index: frames.length, atMs: performance.now() - started, forward, phase, role,
        layer: options.layer, tokenStart: start,
        tokenIds: ids, shape: [count], fullShape, offset: 0,
        dtype: options.dtype, values: null, status: 'captured', reason: null,
        representation: options.representation ?? 'raw', metadata: { originalShape, precision: options.precision ?? null, ...options.metadata } };
      frames.push(frame);
      if (bytes + payloadBytes > limits.activityMaxBytes) {
        frame.status = 'dropped'; frame.reason = 'Activity recording budget reached; inference continues.';
        return;
      }
      try {
        let values;
        if (buffer instanceof Float32Array || buffer instanceof Float64Array) {
          values = buffer.slice(offset, offset + readCount);
        } else {
          if (buffer.size < (offset + readCount) * elementBytes) throw new Error('Activity buffer is smaller than its declared range.');
          // Keep the recorder's resources alive, but submit the producing work
          // before copying. No completion promise can hang after recorder abort.
          options.recorder?.submitPartial();
          const byteOffset = offset * elementBytes;
          const alignedOffset = Math.floor(byteOffset / 4) * 4;
          const prefix = byteOffset - alignedOffset;
          const size = Math.ceil((prefix + readCount * elementBytes) / 4) * 4;
          if (!reader) throw new Error('GPU activity capture has no runtime-owned reader.');
          const raw = await reader(buffer, alignedOffset, size);
          const payload = raw.slice(prefix, prefix + readCount * elementBytes);
          values = options.dtype === 'u32' ? new Uint32Array(payload) : decodeSnapshotData(payload, options.dtype, options);
        }
        if (values.length !== readCount) throw new Error('Incomplete activity capture.');
        frame.values = Array.from(values.slice(0, count), v => Number.isFinite(v) ? v : /** @type {'NaN' | 'Infinity' | '-Infinity'} */ (String(v)));
        if (options.state) {
          /** @type {import('./activity-trace.js').ActivityNumber[]} */
          const summary = [];
          for (let start = 0; start < values.length; start += limits.activitySummaryBlock) {
            let sum = 0;
            const end = Math.min(values.length, start + limits.activitySummaryBlock);
            for (let i = start; i < end; i++) sum += values[i] * values[i];
            const rms = Math.sqrt(sum / (end - start));
            summary.push(Number.isFinite(rms) ? rms : /** @type {'NaN' | 'Infinity' | '-Infinity'} */ (String(rms)));
          }
          frame.summary = { blockSize: limits.activitySummaryBlock, observedValues: values.length, values: summary };
        }
        bytes += payloadBytes;
      } catch (error) {
        frame.status = 'failed'; frame.reason = String(error);
      }
    },
    weightAccess(layer, stage) {
      const roles = /** @type {Record<string, string[]>} */ (stages.weightRoles)[stage];
      const ids = weights.filter(w => w.layer === layer && roles?.includes(w.role)).map(w => w.id);
      if (!ids.length) return;
      void api.capture('weight-access', new Float32Array([1]), { layer, dtype: 'f32', shape: [1, 1],
        metadata: { weightIds: ids, stage, meaning: 'projection dispatch encoded' } });
    },
    async probe(stage, buffer, options) {
      const role = /** @type {Record<string,string>} */ (stages.probes)[stage];
      if (!role) return;
      await api.capture(role, buffer, { ...options, layer: options.layerIdx ?? -1,
        dtype: options.dtype ?? 'f32', shape: [options.numTokens, options.hiddenSize] });
    },
    unavailable(role, reason) { unavailable[role] = reason; },
    setWeights(value) { weights = value; },
    snapshot(includeFrames = true) {
      return { schema: 'doppler.model-activity/v1', frames: includeFrames ? [...frames] : [],
        weights, recordedBytes: bytes, limits: { ...limits }, unavailable: { ...unavailable }, model };
    },
    async logits(values) {
      await api.capture('logits', values, { layer: -1, dtype: 'f32', shape: [1, values.length],
        tokenStart: tokenStart + tokenIds.length - 1, tokenIds: tokenIds.slice(-1) });
    },
  };
  return api;
}
