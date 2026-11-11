import modes from './modes.json' with { type: 'json' };
import config from './config.json' with { type: 'json' };
import { projectFrame, rankProbabilities } from './data.js';

/** @template T @param {T[]} items @param {(item: T) => boolean} predicate */
function lastMatch(items, predicate) {
  for (let i = items.length - 1; i >= 0; i--) if (predicate(items[i])) return items[i];
  return undefined;
}

// Fixed per-layer parameters; they carry no token position of their own.
const PARAMETER_ROLES = new Set(['linear-a-log', 'linear-dt-bias']);

// Token positions at which a view has captured values. null means the view
// draws from layer outputs, which every recorded position has.
/** @type {typeof import('./views.js').viewPositions} */
export function viewPositions(recording, viewId) {
  const roles = positionRoles(viewId);
  if (!roles.length) return null;
  const covered = roles.map(role => {
    const positions = new Set();
    for (const f of recording.trace.activity?.frames ?? []) {
      if (f.role !== role || !hasPicture(f)) continue;
      for (let i = 0; i < span(f); i++) positions.add(f.tokenStart + i);
    }
    return positions;
  });
  return new Set([...covered[0]].filter(position => covered.every(set => set.has(position))));
}

// The capture kinds a recording must keep for one view to be drawable; null
// means every kind (the model map draws from all of them).
/** @type {typeof import('./views.js').viewCaptureRoles} */
export function viewCaptureRoles(viewId) {
  const view = /** @type {{ roles: string[]; recordRoles?: string[] | null } | undefined} */ (modes.find(m => m.id === viewId));
  if (!view) return null;
  return view.recordRoles === undefined ? [...view.roles] : view.recordRoles && [...view.recordRoles];
}

// Captures a view needs per token position; none for views not tied to one position.
/** @param {string} viewId */
function positionRoles(viewId) {
  const view = modes.find(m => m.id === viewId);
  if (!view) return [];
  if (view.clock === 'token') return view.roles.filter(role => !PARAMETER_ROLES.has(role));
  return ['recurrent-memory', 'convolution-history'].includes(view.id) ? [view.id] : [];
}

/** @param {import('./views.js').ActivityFrame} f */
function hasPicture(f) { return f.status === 'captured' && Boolean(f.values || f.summary); }

// A prediction belongs to the one position it was made at.
/** @param {import('./views.js').ActivityFrame} f */
function span(f) { return f.role === 'logits' ? 1 : f.tokenIds.length; }

/** @param {import('./views.js').ActivityFrame} f @param {number} position */
function covers(f, position) { return position >= f.tokenStart && position < f.tokenStart + span(f); }

// Why a view lacks values at a position, read from the recording, and the nearest it can draw.
/** @type {typeof import('./views.js').missingReason} */
export function missingReason(recording, viewId, position, detail = 'overview', layer = -1) {
  const covered = viewPositions(recording, viewId);
  const nearest = covered?.size ? [...covered].reduce((best, p) => Math.abs(p - position) < Math.abs(best - position) ? p : best) : null;
  const activity = recording.trace.activity, frames = activity?.frames ?? [];
  const megabytes = (/** @type {unknown} */ bytes) => Number.isFinite(Number(bytes)) && Number(bytes) > 0 ? ` (${Math.round(Number(bytes) / 2 ** 20)} MB)` : '';
  const limit = megabytes(recording.trace.capturePolicy?.activityMaxBytes);
  const answerStart = Math.min(...recording.trace.frames.filter(f => f.phase === 'decode').map(f => f.tokenStart));
  // What the person recording chose to leave out; absent in recordings made before that choice existed.
  const chosen = activity?.selection ?? null, roles = positionRoles(viewId);
  const keptLayers = chosen?.layers && roles.length ? `This recording kept layer${chosen.layers.length === 1 ? '' : 's'} ${chosen.layers.map(l => l + 1).join(', ')} only.` : '';
  if (keptLayers && layer >= 0 && !chosen?.layers?.includes(layer)) return { kind: 'not-selected', nearest: position, text: keptLayers };
  for (const role of roles) {
    if (chosen?.roles && !chosen.roles.includes(role)) {
      return { kind: 'not-selected', nearest: null, text: 'This recording was set to leave this view out, so nothing was saved for it.' };
    }
    const mine = frames.filter(f => f.role === role);
    const here = mine.filter(f => covers(f, position));
    const failed = here.find(f => f.status === 'failed');
    if (failed) return { kind: 'failed', nearest, text: `Saving this view failed at token ${position}: ${failed.reason}` };
    if (detail === 'raw' && here.some(f => f.status === 'captured' && !f.values && f.summary)) {
      return { kind: 'representation', nearest: position, text: 'Only block summaries were saved at this token, not exact values. Set Detail to Overview to see them.' };
    }
    if (here.some(hasPicture) && !here.some(f => f.status === 'dropped')) continue;
    const dropped = mine.filter(f => f.status === 'dropped');
    if (here.some(f => f.status === 'dropped')) {
      const first = Math.min(...dropped.map(f => f.tokenStart));
      // A view can run out of its own budget while the recording still has room.
      if (here.some(f => f.status === 'dropped' && f.metadata.limit === 'activityRoleMaxBytes')) {
        return { kind: 'dropped', nearest, text: `This view reached its own size limit${megabytes(activity?.coverage?.[role]?.maxBytes)} at token ${first}, so it was not saved after that.` };
      }
      return { kind: 'dropped', nearest, text: `The recording reached its size limit${limit} at token ${first}, so this view was not saved after that.` };
    }
    const range = chosen?.tokens;
    if (range && (position < range.from || (range.to !== null && position >= range.to))) {
      return { kind: 'not-selected', nearest, text: `This recording kept ${range.to === null ? `token ${range.from} onward` : `tokens ${range.from}–${range.to - 1}`} for this view.` };
    }
    const saved = mine.filter(hasPicture);
    if (!saved.length) continue;
    const from = Math.min(...saved.map(f => f.tokenStart)), to = Math.max(...saved.map(f => f.tokenStart + span(f) - 1));
    // A prompt chunk longer than the recording window keeps only its last tokens.
    const windowed = saved.find(f => Number(/** @type {{ originalShape?: number[] }} */ (f.metadata).originalShape?.[0]) > f.fullShape[0]);
    if (windowed && position < windowed.tokenStart) {
      return { kind: 'window', nearest, text: `Saved for the last ${windowed.fullShape[0]} prompt tokens only (tokens ${windowed.tokenStart}–${windowed.tokenStart + windowed.fullShape[0] - 1}).` };
    }
    if (Number.isFinite(answerStart) && position >= answerStart && !mine.some(f => f.phase === 'decode')) {
      return { kind: 'fused', nearest, text: `This stage was never a separate value while the answer was written: the engine ran it inside a fused step, so there was nothing to save. It exists for tokens ${from}–${to}.` };
    }
    return { kind: 'elsewhere', nearest, text: `Saved only for tokens ${from}–${to}.` };
  }
  // Every layer at once: the layers left out stay missing in an otherwise complete picture.
  if (keptLayers && layer < 0) return { kind: 'not-selected', nearest: position, text: keptLayers };
  return { kind: 'none', nearest, text: '' };
}

// What a recording left out and why, from its own coverage counts.
/** @type {typeof import('./views.js').omissionSummary} */
export function omissionSummary(recording) {
  const coverage = Object.entries(recording.trace.activity?.coverage ?? {});
  const total = (/** @type {(c: import('./views.js').ActivityCoverage) => number} */ count) => coverage.reduce((sum, [, c]) => sum + count(c), 0);
  const parts = [];
  const unselected = coverage.filter(([, c]) => c.notSelected.role).map(([role]) => role);
  if (unselected.length) parts.push(`left out by choice: ${unselected.join(', ')}`);
  const layers = total(c => c.notSelected.layer), tokens = total(c => c.notSelected.token), reused = total(c => c.reused);
  if (layers) parts.push(`${layers} captures on layers not selected`);
  if (tokens) parts.push(`${tokens} captures outside the selected tokens`);
  if (reused) parts.push(`${reused} repeats of fixed per-layer values recorded once`);
  const dropped = coverage.filter(([, c]) => c.dropped);
  if (dropped.length) parts.push(`out of budget: ${dropped.map(([role, c]) => `${role} from token ${c.firstDroppedToken}`).join(', ')}`);
  return parts.join(' · ');
}

/** @type {typeof import('./views.js').activityModes} */
export function activityModes(recording) {
  const frames = recording?.trace.activity?.frames ?? [];
  const roles = new Set(frames.filter(f => f.status === 'captured' && f.values).map(f => f.role));
  const weights = recording?.trace.activity?.weights ?? [];
  return modes.map(mode => {
    let reason = mode.roles.every(role => roles.has(role)) ? null : 'This recording has no capture for this view.';
    if (!recording?.trace.frames.length) reason = 'Record an answer or open a saved recording.';
    if (mode.id === 'model-paths' && !Array.isArray(recording?.trace.activity?.model.layerTypes)) reason = 'No layer structure recorded; inspect the captured tensor views instead.';
    if (mode.id === 'weights' && !weights.some(w => w.rms.length)) reason = 'No observed weight tiles in this recording.';
    if (mode.id === 'contributions' && !weights.some(w => w.columnEnergy.length)) reason = 'Missing fixed weights and projection-input pair.';
    if (mode.id === 'contributions' && recording?.trace.identity.adapters?.length) reason = 'Effective adapter weights were not included; base-only products would be incomplete.';
    if (mode.id === 'residency' && !weights.length) reason = 'No loaded-weight inventory in this recording.';
    if (mode.id === 'quantization-error' && weights.some(w => w.error?.length)) reason = null;
    if (!reason && recording && viewPositions(recording, mode.id)?.size === 0) reason = 'This recording has no capture for this view.';
    const kept = recording?.trace.activity?.selection?.roles;
    if (reason && kept && mode.roles.some(role => !kept.includes(role))) reason = 'This recording was set to leave this view out.';
    if (reason) reason = recording?.trace.activity?.unavailable[mode.id] ?? reason;
    return { ...mode, reason };
  });
}

/** @type {typeof import('./views.js').playbackPoints} */
export function playbackPoints(recording, mode, viewId) {
  const view = modes.find(m => m.id === viewId);
  // The timeline only offers positions the view can draw, so it never lands on an empty grid.
  const covered = viewId ? viewPositions(recording, viewId) : null;
  const drawable = (/** @type {{ position: number }} */ point) => !covered || covered.has(point.position);
  const frames = [...recording.trace.frames, ...(recording.trace.activity?.frames ?? [])]
    .filter(f => view?.clock !== 'boundary' || view.id === 'model-paths' || (view.id === 'residency' ? f.role === 'weight-access' : f.role === view.id && hasPicture(/** @type {import('./views.js').ActivityFrame} */ (f))))
    .sort((a, b) => a.atMs - b.atMs);
  if (mode !== 'tokens') return frames.map(f => ({ ...f, position: f.tokenStart + f.tokenIds.length - 1 })).filter(drawable);
  const positions = new Map();
  for (const f of frames) {
    for (let i = 0; i < f.tokenIds.length; i++) positions.set(f.tokenStart + i,
      { ...f, position: f.tokenStart + i });
  }
  return [...positions.values()].filter(drawable).sort((a, b) => a.position - b.position);
}

/** @type {typeof import('./views.js').assertMatchedRuns} */
export function assertMatchedRuns(a, b) {
  if (!a.trace.identity.manifestHash || a.trace.identity.manifestHash !== b.trace.identity.manifestHash
    || JSON.stringify(a.trace.identity.adapters ?? []) !== JSON.stringify(b.trace.identity.adapters ?? [])
    || JSON.stringify(a.trace.geometry) !== JSON.stringify(b.trace.geometry)) throw new Error('Comparison requires the same exact artifact, adapters and layer geometry.');
  const ids = (/** @type {import('./data.js').ActivationRecording} */ r) => [...(r.inputTokenIds ?? []), ...(r.generatedTokenIds ?? [])];
  const aa = ids(a), bb = ids(b);
  if (!aa.length || !bb.length) throw new Error('Comparison requires recorded exact input and generated token IDs.');
  const shared = Math.min(aa.length, bb.length);
  if (aa.slice(0, shared).some((id, i) => id !== bb[i])) throw new Error('Token prefixes differ. Replay the same exact token IDs before comparing activity.');
}

/** @param {import('./views.js').ActivityFrame | undefined} f @param {number} position */
function tokenRow(f, position) {
  if (!f?.values || f.status !== 'captured') return [];
  const width = f.fullShape[1], row = position - f.tokenStart;
  if (row < 0 || row >= f.fullShape[0] || !width) return [];
  return f.values.slice(row * width, (row + 1) * width);
}

/** @param {(number | string | undefined)[]} values */
function softmax(values) {
  if (!values.length || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) return values.map(() => undefined);
  let max = -Infinity;
  for (const value of values) max = Math.max(max, Number(value));
  const exp = values.map(v => Math.exp(Number(v) - max)), sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}

/** @type {typeof import('./views.js').attentionRow} */
export function attentionRow(frames, query, position, head) {
  const m = query.metadata, heads = Number(m.heads), kvHeads = Number(m.kvHeads), dim = Number(m.headDim);
  const length = Number(m.kvLength), scale = Number(m.scale), softcap = Number(m.softcap), window = Number(m.slidingWindow);
  const q = tokenRow(query, position).slice(head * dim, (head + 1) * dim);
  const values = new Array(length).fill(undefined);
  if (q.length !== dim || !heads || !kvHeads || heads % kvHeads || !Number.isFinite(scale)) return values;
  const kh = Math.floor(head / (heads / kvHeads));
  const keys = frames.filter(f => f.role === 'kv-key' && f.layer === query.layer && f.atMs <= query.atMs && f.status === 'captured');
  let max = -Infinity;
  for (let k = 0; k < length; k++) {
    if ((m.causal && k > position) || (window > 0 && k < position - window + 1)) { values[k] = -Infinity; continue; }
    const source = lastMatch(keys, f => k >= f.tokenStart && k < f.tokenStart + f.tokenIds.length);
    const key = tokenRow(source, k).slice(kh * dim, (kh + 1) * dim);
    if (key.length !== dim || [...key, ...q].some(v => typeof v !== 'number' || !Number.isFinite(v))) return values.map(() => undefined);
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += Number(q[d]) * Number(key[d]);
    let score = dot * scale;
    if (softcap > 0) score = softcap * Math.tanh(score / softcap);
    values[k] = score; max = Math.max(max, score);
  }
  if (!Number.isFinite(max)) return values.map(() => undefined);
  const exp = values.map(v => Math.exp(v - max)), sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}

/** Rectangular head-by-key view at the shared playback position.
 * @type {typeof import('./views.js').attentionGridView} */
export function attentionGridView(recording, atMs, position, layer = -1) {
  const frames = (recording.trace.activity?.frames ?? []).filter(f => f.atMs <= atMs);
  const queries = [...new Map(frames.filter(f => f.role === 'attention-query' && (layer < 0 || f.layer === layer)
    && position >= f.tokenStart && position < f.tokenStart + f.tokenIds.length).map(f => [f.layer, f])).values()]
    .sort((a, b) => a.layer - b.layer);
  const width = Math.max(0, ...queries.map(q => Number(q.metadata.kvLength)));
  const height = queries.reduce((sum, q) => sum + Number(q.metadata.heads), 0);
  const why = missingReason(recording, 'attention-query', position, 'overview', layer);
  if (!width || !height || width * height > config.maxDisplayValues) return { values: [], width: 0, rows: [], position,
    reason: width * height > config.maxDisplayValues ? 'Attention exceeds the display budget. Select a layer to narrow the grid.' : why.text || 'No attention queries saved at this position.', nearest: why.nearest };
  const rows = queries.flatMap(q => Array.from({ length: Number(q.metadata.heads) }, (_, head) => ({ layer: q.layer, head })));
  const values = queries.flatMap(q => Array.from({ length: Number(q.metadata.heads) }, (_, head) => {
    const row = attentionRow(frames, q, position, head);
    return Array.from({ length: width }, (_, key) => key < row.length ? row[key] : undefined);
  }).flat());
  return { values, width, rows, position, reason: values.some(v => v !== undefined) ? '' : why.text || 'Queries or keys were not saved at this position.', nearest: why.nearest };
}

/** @type {typeof import('./views.js').activityView} */
export function activityView(recording, selection) {
  const { mode, atMs, position, layer, detail, weightId, baseline } = selection;
  const trace = recording.trace, activity = trace.activity;
  const frames = (activity?.frames ?? []).filter(f => f.atMs <= atMs);
  const latest = (/** @type {string} */ role, /** @type {number} */ l) => lastMatch(frames, f => f.role === role && f.layer === l
    && position >= f.tokenStart && position < f.tokenStart + f.tokenIds.length);
  const weights = (activity?.weights ?? []).filter(w => layer < 0 || w.layer === layer);
  /** @type {(number | string | undefined)[]} */
  let values = [];
  /** @type {{start:number; end:number; id:string; count:number; blockSize:number}[]} */
  const segments = [];
  let underlyingValues = 0, extent = 0, offset = 0, quantity = mode, representation = 'raw', description = '', valuesPerScalar = 1;
  let distribution = null;
  const tile = (/** @type {number} */ size, /** @type {(index: number) => number | string | undefined} */ get) => {
    extent = size;
    offset = Math.min(Math.max(0, Math.floor(selection.page)) * config.maxDisplayValues,
      Math.max(0, Math.floor((size - 1) / config.maxDisplayValues) * config.maxDisplayValues));
    values = Array.from({ length: Math.min(size - offset, config.maxDisplayValues) }, (_, i) => get(offset + i));
  };
  const output = (/** @type {import('./data.js').ActivationRecording} */ r, /** @type {number} */ pos, /** @type {number} */ time) => {
    const index = lastMatch(r.trace.frames, f => f.atMs <= time && pos >= f.tokenStart && pos < f.tokenStart + f.tokenIds.length)?.index ?? -1;
    if (index < 0) return new Array(r.trace.geometry.layers * r.trace.geometry.width);
    return projectFrame(r.trace, index, pos).values;
  };
  if (mode === 'layer-output' || mode === 'difference') {
    values = output(recording, position, atMs); quantity = 'layer-output values';
    if (mode === 'difference') {
      if (baseline) assertMatchedRuns(recording, baseline);
      const other = baseline ? output(baseline, position, Infinity) : output(recording, position - 1, atMs);
      values = values.map((v, i) => typeof v === 'number' && typeof other[i] === 'number' ? v - Number(other[i]) : undefined);
      representation = 'reconstructed'; quantity = 'signed activation difference';
      description = baseline ? 'Current minus the matched saved run, at the same token position.' : 'Current token minus the previous token position.';
    } else description = 'A composite of sequential layer captures, not simultaneous state.';
    if (layer >= 0) values = values.slice(layer * trace.geometry.width, (layer + 1) * trace.geometry.width);
  } else if (mode.startsWith('ffn-')) {
    const width = Number(activity?.model.intermediateSize), layers = layer < 0 ? Array.from({ length: trace.geometry.layers }, (_, i) => i) : [layer];
    for (const l of layers) {
      const row = tokenRow(latest(mode, l), position);
      values.push(...Array.from({ length: width }, (_, i) => row[i]));
    }
    quantity = mode === 'ffn-product' ? 'activated gate × up-projection values' : `${mode} values`;
    description = 'Only materialized stages are captured. Fused-away intermediates and positions outside the recorded window are missing.';
  } else if (mode === 'weights' || mode === 'residency' || mode === 'contributions') {
    const selected = weightId ? weights.filter(w => w.id === weightId) : weights;
    if (mode === 'residency') {
      const accesses = frames.filter(f => f.role === 'weight-access' && position >= f.tokenStart && position < f.tokenStart + f.tokenIds.length);
      const accessed = new Set(accesses.flatMap(f => /** @type {string[]} */ (f.metadata.weightIds)));
      values = selected.map(w => accessed.has(w.id) ? 1 : w.resident ? 0.25 : 0); quantity = 'resident tensor and observed projection binding';
      description = 'One scalar per tensor: 1 = down-projection dispatch encoded for this position, 0.25 = resident at capture start, 0 = not resident. Other operator accesses, uploads and hardware cache hits are unobserved. This is not numerical importance.';
    } else if (mode === 'weights') {
      underlyingValues = selected.reduce((n, w) => n + (w.shape.length ? w.shape.reduce((a, b) => a * b, 1) : 0), 0);
      if (detail === 'raw') {
        const w = selected.find(w => w.raw.length);
        if (w) { underlyingValues = w.shape.reduce((a, b) => a * b, 1); values = [...w.raw]; description = `${w.id}: saved exact window begins at value ${w.rawOffset}. Other connections require the matching weight bytes.`; }
        else description = 'No exact weight window saved for this tensor; use its overview.';
        quantity = 'dequantized stored weight';
      } else {
        let blocks = 0;
        for (const w of selected) {
          if (!w.shape.length) continue;
          const count = w.shape.reduce((a, b) => a * b, 1), size = Math.ceil(count / w.blockSize);
          segments.push({ start: blocks, end: blocks + size, id: w.id, count, blockSize: w.blockSize });
          blocks += size;
        }
        const byId = new Map(selected.map(w => [w.id, w]));
        let segmentIndex = 0;
        tile(blocks, i => {
          while (segmentIndex < segments.length && i >= segments[segmentIndex].end) segmentIndex++;
          const segment = segments[segmentIndex];
          return segment ? byId.get(segment.id)?.rms[i - segment.start] : undefined;
        });
        quantity = 'weight RMS per block'; representation = 'aggregated';
        valuesPerScalar = Math.max(1, ...selected.filter(w => w.shape.length).map(w => w.blockSize));
        description = `${selected.filter(w => w.status === 'captured').length}/${selected.length} loaded tensors observed. Blocks never cross tensor boundaries; final blocks may be shorter. ${selected.filter(w => !w.shape.length).length} tensors have no recorded shape and are excluded from the atlas. Known-size missing tensors retain their full block extent.`;
      }
    } else {
      const w = selected.find(w => w.columnEnergy.length);
      description = 'No weight/input pair captured for this tensor or layer.';
      if (w) {
        const input = tokenRow(latest('ffn-down-input', Number(w.layer)), position);
        const [rows, cols] = w.shape;
        underlyingValues = rows * cols;
        if (detail === 'raw') {
          tile(w.raw.length, i => typeof w.raw[i] === 'number' && typeof input[(w.rawOffset + i) % cols] === 'number' ? w.raw[i] * Number(input[(w.rawOffset + i) % cols]) : undefined);
          quantity = 'weight × projection input'; representation = 'reconstructed';
          description = `${w.id}: exact mathematical products in the saved weight window. Not kernel-rounded products or causal attribution. Other connections require matching weights.`;
        } else {
          tile(w.columnEnergy.length, i => {
            const x = input[i % cols], count = Math.min(w.rowGroupSize, rows - Math.floor(i / cols) * w.rowGroupSize);
            return typeof x === 'number' ? Math.sqrt(w.columnEnergy[i] * x * x / count) : undefined;
          });
          quantity = 'RMS connection product per output-row group'; representation = 'aggregated'; valuesPerScalar = w.rowGroupSize;
          description = `${w.id}: precomputed sum of squared weights × input², grouped over up to ${w.rowGroupSize} output rows. No per-token weight matrix is materialized.`;
        }
      }
    }
  } else if (mode === 'recurrent-memory' || mode === 'convolution-history') {
    const sources = frames.filter(f => f.role === mode && (layer < 0 || f.layer === layer)
      && position >= f.tokenStart && position < f.tokenStart + f.tokenIds.length);
    const latestByLayer = [...new Map(sources.map(f => [f.layer, f])).values()].sort((a, b) => a.layer - b.layer);
    const boundaries = [...new Set(latestByLayer.map(f => `${f.phase} positions ${f.tokenStart}–${f.tokenStart + f.tokenIds.length - 1}`))].join(', ');
    const fullCount = latestByLayer.reduce((n, f) => n + f.fullShape.reduce((a, b) => a * b, 1), 0);
    underlyingValues = fullCount;
    if (detail === 'overview' && latestByLayer.some(f => f.summary)) {
      valuesPerScalar = latestByLayer.find(f => f.summary)?.summary?.blockSize ?? 1;
      let blocks = 0;
      for (const f of latestByLayer) {
        const count = f.fullShape.reduce((a, b) => a * b, 1), blockSize = f.summary?.blockSize ?? valuesPerScalar;
        segments.push({start: blocks, end: blocks + Math.ceil(count / blockSize), id: String(f.layer), count, blockSize});
        blocks += Math.ceil(count / blockSize);
      }
      tile(blocks, i => {
        const n = segments.findIndex(s => i >= s.start && i < s.end);
        return n < 0 ? undefined : latestByLayer[n].summary?.values[i - segments[n].start];
      });
      representation = 'aggregated'; quantity = `${mode} · block RMS`;
      description = `${fullCount.toLocaleString()} full-state values across ${latestByLayer.length} layers; summaries cover ${latestByLayer.reduce((n, f) => n + (f.summary?.observedValues ?? 0), 0).toLocaleString()}. State after ${boundaries}; prefill does not expose intermediate per-token states.`;
    } else {
      values = latestByLayer.flatMap(f => f.values ?? new Array(f.shape[0]).fill(undefined));
      quantity = mode === 'recurrent-memory' ? 'recurrent state values' : 'convolution history values';
      description = `Saved raw prefixes (${values.length.toLocaleString()} of ${fullCount.toLocaleString()} full-state values). State after ${boundaries}, not intermediate prefill positions. Unrecorded state needs a new capture.`;
    }
  } else if (mode === 'kv-new' || mode === 'kv-cache') {
    const layers = [...new Set(frames.filter(f => f.role === 'kv-key' && (layer < 0 || layer === f.layer)).map(f => f.layer))];
    const positions = mode === 'kv-cache' ? position + 1 : 1;
    const width = frames.find(f => f.role === 'kv-key')?.fullShape[1] ?? 0;
    const rows = new Map();
    tile(layers.length * positions * width * 2, i => {
      const l = layers[Math.floor(i / (positions * width * 2))], within = i % (positions * width * 2);
      const pos = mode === 'kv-cache' ? Math.floor(within / (width * 2)) : position;
      const role = within % (width * 2) < width ? 'kv-key' : 'kv-value';
      const key = `${l}/${role}/${pos}`;
      if (!rows.has(key)) {
        const f = lastMatch(frames, f => f.role === role && f.layer === l && pos >= f.tokenStart && pos < f.tokenStart + f.tokenIds.length);
        rows.set(key, tokenRow(f, pos));
      }
      return rows.get(key)[within % width];
    });
    quantity = 'stored keys and values'; description = 'Append-only cache writes, ordered layer → position → keys then values. Missing entries remain dark; unused capacity is excluded.';
  } else if (mode === 'attention-query' || mode === 'attention-prefill') {
    const queries = frames.filter(f => f.role === 'attention-query' && (layer < 0 || layer === f.layer)
      && (mode === 'attention-prefill' ? f.phase === 'prefill' : position >= f.tokenStart && position < f.tokenStart + f.tokenIds.length));
    const byLayer = new Map();
    for (const q of queries) byLayer.set(q.layer, q);
    const groups = [...byLayer.values()].map(q => ({
      layer: q.layer, heads: Number(q.metadata.heads), length: Number(q.metadata.kvLength),
      rows: mode === 'attention-prefill' ? Math.max(...queries.filter(f => f.layer === q.layer).map(f => f.tokenStart + f.tokenIds.length)) : 1,
    }));
    /** @type {number[]} */
    const ends = [];
    let total = 0;
    for (const g of groups) { total += g.rows * g.heads * g.length; ends.push(total); }
    const cache = new Map();
    tile(total, i => {
      const group = ends.findIndex(end => i < end), g = groups[group];
      const local = i - (group ? ends[group - 1] : 0), row = Math.floor(local / g.length);
      const head = row % g.heads, pos = mode === 'attention-prefill' ? Math.floor(row / g.heads) : position;
      const key = `${g.layer}/${pos}/${head}`;
      if (!cache.has(key)) {
        const q = lastMatch(queries, f => f.layer === g.layer && pos >= f.tokenStart && pos < f.tokenStart + f.tokenIds.length);
        cache.set(key, q ? attentionRow(frames, q, pos, head) : []);
      }
      const probabilities = cache.get(key), k = local % g.length;
      return k >= probabilities.length && probabilities.length ? 0 : probabilities[k];
    });
    quantity = 'attention probabilities'; representation = 'reconstructed';
    description = `${groups.length} captured layer(s), ordered layer → query → head → key. Prompt chunks share global position coordinates. CPU double reconstruction from captured Q and stored K, with recorded scale, mask and softcap; future masked positions are zero. Not kernel bit-exact or causal attribution.`;
  } else if (mode === 'linear-gates') {
    const layers = [...new Set(frames.filter(f => f.role === 'linear-a' && (layer < 0 || f.layer === layer)).map(f => f.layer))];
    for (const l of layers) {
      const a = tokenRow(latest('linear-a', l), position), b = tokenRow(latest('linear-b', l), position);
      const alog = lastMatch(frames, f => f.role === 'linear-a-log' && f.layer === l)?.values;
      const bias = lastMatch(frames, f => f.role === 'linear-dt-bias' && f.layer === l)?.values;
      for (let h = 0; h < a.length; h++) {
        const x = Number(a[h]) + Number(bias?.[h]);
        const keep = Math.exp(-Math.exp(Number(alog?.[h])) * (Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)))));
        const update = 1 / (1 + Math.exp(-Number(b[h])));
        values.push(Number.isFinite(keep) ? keep : undefined, Number.isFinite(update) ? update : undefined);
      }
    }
    quantity = 'retention factor, update strength per head'; representation = 'reconstructed';
    description = 'Recomputed from captured projections and parameters; alternating retention exp(log-decay) and update sigmoid values. Not a claim of kernel bit-exact arithmetic.';
  } else if (mode === 'logits' || mode === 'probabilities') {
    const f = lastMatch(frames, f => f.role === 'logits' && f.tokenStart === position);
    values = f?.values ?? []; quantity = mode === 'logits' ? 'next-token logits' : 'full-vocabulary softmax probability';
    if (mode === 'probabilities') { values = softmax(values); distribution = rankProbabilities(values); representation = 'reconstructed'; }
    description = mode === 'probabilities'
      ? 'Softmax probabilities over the whole vocabulary, in token-ID order, before sampling filters (temperature, top-k, top-p) are applied. Dark pixels are real probabilities near zero, not missing data; a position with no saved prediction is checkered.'
      : 'Raw next-token scores over the whole vocabulary, in token-ID order, before softmax and before sampling filters. A position with no saved prediction is checkered.';
  } else if (mode === 'expert-routing') {
    const f = lastMatch(frames, f => f.role === 'expert-indices' && (layer < 0 || layer === f.layer)
      && position >= f.tokenStart && position < f.tokenStart + f.tokenIds.length);
    if (f) {
      values = new Array(Number(f.metadata.numExperts)).fill(0);
      const indices = tokenRow(f, position), scores = tokenRow(latest('expert-weights', f.layer), position);
      if (!indices.length || scores.length !== indices.length) values.fill(undefined);
      else indices.forEach((index, i) => { values[Number(index)] = scores[i]; });
      description = `Layer ${f.layer + 1}; selected experts carry their routing weights. Zero means not selected, not unused model knowledge.`;
      quantity = 'selected-expert routing weight';
    }
  } else if (mode === 'quantization-error') {
    const w = weightId ? weights.find(w => w.id === weightId && w.error?.length) : weights.find(w => w.error?.length);
    description = 'No source-reference pair captured for this tensor or layer.';
    if (w) {
      values = [...(w.error ?? [])];
      underlyingValues = w.shape.reduce((a, b) => a * b, 1);
      quantity = 'stored head minus tied embedding reference'; representation = 'reconstructed';
      description = `${w.id} (${w.dtype}) minus ${w.referenceId} (${w.referenceDtype}) from the same pinned artifact, whose contract ties these weights. Exact saved window only; reference precision is not original F32 precision.`;
    }
  }
  if (!extent) extent = values.length;
  if (values.length > config.maxDisplayValues) { const all = values; tile(all.length, i => all[i]); }
  const range = mode === 'residency' ? 'residency'
    : ['attention-query', 'attention-prefill', 'probabilities', 'linear-gates', 'expert-routing'].includes(mode) ? 'unit'
    : representation === 'aggregated' ? 'positive' : 'signed';
  return { values, extent, offset, range, segments, underlyingValues: underlyingValues || extent, quantity, representation, description, valuesPerScalar, distribution };
}
