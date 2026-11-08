import modes from './modes.json' with { type: 'json' };
import config from './config.json' with { type: 'json' };
import { projectFrame } from './data.js';

/** @template T @param {T[]} items @param {(item: T) => boolean} predicate */
function lastMatch(items, predicate) {
  for (let i = items.length - 1; i >= 0; i--) if (predicate(items[i])) return items[i];
  return undefined;
}

/** @type {typeof import('./views.js').activityModes} */
export function activityModes(recording) {
  const frames = recording?.trace.activity?.frames ?? [];
  const roles = new Set(frames.filter(f => f.status === 'captured').map(f => f.role));
  const weights = recording?.trace.activity?.weights ?? [];
  return modes.map(mode => {
    let reason = mode.roles.every(role => roles.has(role)) ? null : 'This recording has no capture for this view.';
    if (!recording?.trace.frames.length) reason = 'Record an answer or open a saved recording.';
    if (mode.id === 'weights' && !weights.some(w => w.rms.length)) reason = 'No observed weight tiles in this recording.';
    if (mode.id === 'contributions' && !weights.some(w => w.columnEnergy.length)) reason = 'Missing fixed weights and projection-input pair.';
    if (mode.id === 'contributions' && recording?.trace.identity.adapters?.length) reason = 'Effective adapter weights were not included; base-only products would be incomplete.';
    if (mode.id === 'residency' && !weights.length) reason = 'No loaded-weight inventory in this recording.';
    if (mode.id === 'quantization-error' && weights.some(w => w.error?.length)) reason = null;
    if (reason) reason = recording?.trace.activity?.unavailable[mode.id] ?? reason;
    return { id: mode.id, label: mode.label, reason };
  });
}

/** @type {typeof import('./views.js').playbackPoints} */
export function playbackPoints(recording, mode) {
  const frames = [...recording.trace.frames, ...(recording.trace.activity?.frames ?? [])]
    .sort((a, b) => a.atMs - b.atMs);
  if (mode !== 'tokens') return frames.map(f => ({ ...f, position: f.tokenStart + f.tokenIds.length - 1 }));
  const positions = new Map();
  for (const f of frames) {
    for (let i = 0; i < f.tokenIds.length; i++) positions.set(f.tokenStart + i,
      { ...f, position: f.tokenStart + i });
  }
  return [...positions.values()].sort((a, b) => a.position - b.position);
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
    if (mode === 'probabilities') { values = softmax(values); representation = 'reconstructed'; }
    description = 'Vocabulary token-ID order. These are scores before sampling filters; a missing position has no captured prediction.';
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
  return { values, extent, offset, range, segments, underlyingValues: underlyingValues || extent, quantity, representation, description, valuesPerScalar };
}
