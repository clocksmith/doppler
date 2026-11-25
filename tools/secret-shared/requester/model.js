/**
 * Qwen 3.5 (Gated DeltaNet + gated full attention) for the secret-shared requester, in plain JS on typed arrays.
 *
 * Every projection goes to `linear.run(groupId, x, n)` (the share operators, or a plaintext stand-in); everything
 * else runs here: RMSNorm, the DeltaNet causal convolution, recurrence and gated norm, q/k norms, partial RoPE,
 * attention over an F16 KV cache with F32 arithmetic, the output gate, SiLU, residual adds. The equations are
 * Transformers' Qwen 3.5 (as in tools/qwen35-deployed-reference.py), with the deployed F16 KV storage. Sums run in
 * float64 and values are stored as float32. The DeltaNet recurrence runs per position for prefill as well; it is the
 * same function as the chunked form Transformers uses for prefill.
 *
 * The forward pass takes k positions at once (prefill, or k speculative positions to verify in one round per group)
 * and returns logits for the last `keep` of them. With { speculative: true } it keeps what it needs to roll back:
 * `commit(accepted)` keeps the first `accepted` positions and restores the DeltaNet state of the rest without a round.
 *
 * Weights come from `source.tensor(name)` -> { dtype, shape, bytes } and the embedding from
 * `source.embeddingRow(id, dst)`, so a Node file reader and a browser loader can both feed it.
 */
import { float16ToFloat32, float32ToFloat16 } from '../../../src/formats/float16.js';
import { deltaRuleHeads } from './delta-rule.js';

const F16 = new Float32Array(65536);
for (let i = 0; i < 65536; i++) F16[i] = float16ToFloat32(i);

function toF32(t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  const view = new DataView(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = t.dtype === 'F16' ? F16[view.getUint16(2 * i, true)] : view.getFloat32(4 * i, true);
  return out;
}

const sigmoid = v => 1 / (1 + Math.exp(-v));
const silu = v => v / (1 + Math.exp(-v));
const softplus = v => (v > 20 ? v : Math.log1p(Math.exp(v)));

/** Model shapes from the RDRR manifest and the source config's text_config. */
export function configFrom(manifest, source) {
  const a = manifest.architecture, t = source.text_config ?? source;
  const rope = t.rope_parameters ?? {};
  return {
    layers: a.numLayers, hidden: a.hiddenSize, inter: a.intermediateSize, heads: a.numAttentionHeads, kvHeads: a.numKeyValueHeads,
    headDim: a.headDim, vocab: a.vocabSize, linKH: a.linearNumKeyHeads, linVH: a.linearNumValueHeads, linDK: a.linearKeyHeadDim,
    linDV: a.linearValueHeadDim, convK: a.linearConvKernelDim, layerTypes: manifest.inference.layerPattern.layerTypes.slice(0, a.numLayers),
    eps: t.rms_norm_eps, ropeTheta: rope.rope_theta, rotaryDim: Math.floor(a.headDim * (rope.partial_rotary_factor ?? 1)),
    eos: [manifest.eos_token_id].flat()
  };
}

export class QwenRequesterModel {
  constructor(config, source, linear, { parallel = null } = {}) {
    this.c = config; this.source = source; this.linear = linear; this.parallel = parallel;
    // With a worker pool, the DeltaNet state and per-layer inputs live in shared memory and value heads split across threads.
    this.alloc = parallel ? parallel.alloc : n => new Float32Array(n);
    this.dims = { dk: config.linDK, dv: config.linDV, VH: config.linVH, KH: config.linKH, eps: config.eps };
    const c = config, p = 'model.language_model.';
    const plusOne = w => { for (let i = 0; i < w.length; i++) w[i] = Math.fround(1 + w[i]); return w; };
    this.layers = c.layerTypes.map((kind, l) => {
      const L = `${p}layers.${l}.`;
      const layer = { kind, inNorm: plusOne(toF32(source.tensor(L + 'input_layernorm.weight'))), postNorm: plusOne(toF32(source.tensor(L + 'post_attention_layernorm.weight'))) };
      if (kind === 'linear_attention') {
        layer.conv = toF32(source.tensor(L + 'linear_attn.conv1d.weight'));
        layer.negA = toF32(source.tensor(L + 'linear_attn.A_log')).map(v => -Math.exp(v));
        layer.dtBias = toF32(source.tensor(L + 'linear_attn.dt_bias'));
        layer.gnorm = toF32(source.tensor(L + 'linear_attn.norm.weight'));
      } else {
        layer.qNorm = plusOne(toF32(source.tensor(L + 'self_attn.q_norm.weight')));
        layer.kNorm = plusOne(toF32(source.tensor(L + 'self_attn.k_norm.weight')));
      }
      return layer;
    });
    this.finalNorm = plusOne(toF32(source.tensor(p + 'norm.weight')));
    // RoPE inverse frequencies as Transformers computes them, in float32.
    const half = c.rotaryDim / 2;
    this.invFreq = new Float32Array(half);
    for (let i = 0; i < half; i++) this.invFreq[i] = Math.fround(1 / Math.fround(Math.pow(c.ropeTheta, Math.fround((2 * i) / c.rotaryDim))));
    this.held = { tensors: source.heldTensors?.() ?? null, bytes: source.heldBytes?.() ?? null };
  }

  newState() {
    const c = this.c, convDim = 2 * c.linKH * c.linDK + c.linVH * c.linDV;
    return {
      pos: 0,
      conv: this.layers.map(L => (L.kind === 'linear_attention' ? new Float32Array((c.convK - 1) * convDim) : null)),
      rec: this.layers.map(L => (L.kind === 'linear_attention' ? this.alloc(c.linVH * c.linDK * c.linDV) : null)),
      k: this.layers.map(L => (L.kind === 'linear_attention' ? null : { data: new Uint16Array(0), len: 0 })),
      v: this.layers.map(L => (L.kind === 'linear_attention' ? null : { data: new Uint16Array(0), len: 0 })),
      snapshot: null
    };
  }

  rmsnorm(x, n, width, w, out = new Float32Array(n * width), groupWidth = width) {
    const eps = this.c.eps;
    for (let r = 0; r < (n * width) / groupWidth; r++) {
      const o = r * groupWidth;
      let s = 0;
      for (let i = 0; i < groupWidth; i++) s += x[o + i] * x[o + i];
      const inv = Math.fround(1 / Math.sqrt(Math.fround(Math.fround(s / groupWidth) + eps)));
      for (let i = 0; i < groupWidth; i++) out[o + i] = Math.fround(x[o + i] * inv) * w[i];
    }
    return out;
  }

  /** Runs positions `tokens` from state.pos; returns Float32Array [keep * vocab] logits for the last `keep`. */
  async forward(state, tokens, { keep = 1, speculative = false } = {}) {
    const c = this.c, n = tokens.length, H = c.hidden;
    if (speculative) {
      state.snapshot = { pos: state.pos, n, conv: state.conv.map(a => a && a.slice()), rec: state.rec.map(a => a && a.slice()), steps: this.layers.map(() => []) };
    } else state.snapshot = null;
    const x = new Float32Array(n * H);
    for (let t = 0; t < n; t++) this.source.embeddingRow(tokens[t], x.subarray(t * H, (t + 1) * H));
    for (let l = 0; l < c.layers; l++) {
      const L = this.layers[l];
      const h = this.rmsnorm(x, n, H, L.inNorm);
      const o = L.kind === 'linear_attention' ? await this.deltaNet(l, L, h, n, state) : await this.attention(l, L, h, n, state);
      for (let i = 0; i < x.length; i++) x[i] += o[i];
      const h2 = this.rmsnorm(x, n, H, L.postNorm);
      const gu = await this.linear.run(`L${l}.up`, h2, n);
      const I = c.inter, act = new Float32Array(n * I);
      for (let t = 0; t < n; t++) for (let i = 0; i < I; i++) act[t * I + i] = Math.fround(silu(gu[t * 2 * I + i])) * gu[t * 2 * I + I + i];
      const d = await this.linear.run(`L${l}.down`, act, n);
      for (let i = 0; i < x.length; i++) x[i] += d[i];
    }
    const last = this.rmsnorm(x.subarray((n - keep) * H), keep, H, this.finalNorm);
    const logits = await this.linear.run('head', last, keep);
    state.pos += n;
    return logits;
  }

  /** After a speculative forward of k positions, keeps the first `accepted` and rolls the rest back, without a round. */
  commit(state, accepted) {
    const s = state.snapshot;
    if (!s) throw Error('commit needs a speculative forward');
    state.snapshot = null;
    if (accepted === s.n) return;
    const c = this.c, convDim = 2 * c.linKH * c.linDK + c.linVH * c.linDV, kvw = c.kvHeads * c.headDim;
    for (let l = 0; l < c.layers; l++) {
      if (this.layers[l].kind === 'linear_attention') {
        state.conv[l].set(s.conv[l]); state.rec[l].set(s.rec[l]);
        for (let t = 0; t < accepted; t++) {
          const st = s.steps[l][t];
          deltaRuleHeads(state.rec[l], st.q, st.k, st.v, st.g, st.beta, null, null, new Float32Array(c.linVH * c.linDV), 1, this.dims, 0, c.linVH);
          shiftConv(state.conv[l], st.mixed, convDim, c.convK);
        }
      } else {
        state.k[l].len = state.v[l].len = (s.pos + accepted) * kvw;
      }
    }
    state.pos = s.pos + accepted;
  }

  async deltaNet(l, L, h, n, state) {
    const c = this.c, kd = c.linKH * c.linDK, vd = c.linVH * c.linDV, convDim = 2 * kd + vd, VH = c.linVH;
    const r = await this.linear.run(`L${l}.in`, h, n);
    const width = convDim + vd + 2 * VH;
    const alloc = this.alloc;
    const q = alloc(n * kd), k = alloc(n * kd), v = alloc(n * vd), z = alloc(n * vd), g = alloc(n * VH), beta = alloc(n * VH), core = alloc(n * vd);
    const conv = state.conv[l], K = c.convK;
    const qScale = Math.fround(1 / Math.sqrt(c.linDK));
    const post = new Float32Array(convDim);
    for (let t = 0; t < n; t++) {
      const row = r.subarray(t * width, (t + 1) * width);
      const mixed = row.subarray(0, convDim);
      // Causal depthwise convolution over the last K inputs (K - 1 kept in state), then SiLU.
      for (let ch = 0; ch < convDim; ch++) {
        let s = 0;
        for (let j = 0; j < K - 1; j++) s += L.conv[ch * K + j] * conv[j * convDim + ch];
        s += L.conv[ch * K + K - 1] * mixed[ch];
        post[ch] = silu(Math.fround(s));
      }
      shiftConv(conv, mixed, convDim, K);
      const qt = q.subarray(t * kd, (t + 1) * kd), kt = k.subarray(t * kd, (t + 1) * kd);
      qt.set(post.subarray(0, kd)); kt.set(post.subarray(kd, 2 * kd)); v.set(post.subarray(2 * kd), t * vd);
      l2norm(qt, c.linDK, qScale); l2norm(kt, c.linDK, 1);
      z.set(row.subarray(convDim, convDim + vd), t * vd);
      for (let hv = 0; hv < VH; hv++) {
        beta[t * VH + hv] = sigmoid(row[convDim + vd + hv]);
        g[t * VH + hv] = L.negA[hv] * Math.fround(softplus(Math.fround(row[convDim + vd + VH + hv] + L.dtBias[hv])));
      }
      if (state.snapshot) state.snapshot.steps[l].push({ q: qt.slice(), k: kt.slice(), v: v.slice(t * vd, (t + 1) * vd), g: g.slice(t * VH, (t + 1) * VH), beta: beta.slice(t * VH, (t + 1) * VH), mixed: mixed.slice() });
    }
    const d = this.dims;
    // A state copied with slice() (for example to decode several ways from one prefill) is no longer shared memory;
    // move it back so the workers update the caller's state, not a structured clone of it.
    if (this.parallel && !(state.rec[l].buffer instanceof SharedArrayBuffer)) { const shared = this.alloc(state.rec[l].length); shared.set(state.rec[l]); state.rec[l] = shared; }
    if (this.parallel) await this.parallel.deltaRule({ S: state.rec[l], q, k, v, g, beta, z, gnorm: L.gnorm, out: core, n, d });
    else deltaRuleHeads(state.rec[l], q, k, v, g, beta, z, L.gnorm, core, n, d, 0, VH);
    return this.linear.run(`L${l}.out`, core, n);
  }

  async attention(l, L, h, n, state) {
    const c = this.c, hd = c.headDim, NH = c.heads, KV = c.kvHeads, rot = c.rotaryDim, half = rot / 2;
    const r = await this.linear.run(`L${l}.in`, h, n);
    const width = NH * hd * 2 + 2 * KV * hd;
    const q = new Float32Array(n * NH * hd), gate = new Float32Array(n * NH * hd);
    const kNew = new Uint16Array(n * KV * hd), vNew = new Uint16Array(n * KV * hd);
    const kTmp = new Float32Array(KV * hd);
    const cos = new Float32Array(half), sin = new Float32Array(half);
    for (let t = 0; t < n; t++) {
      const row = r.subarray(t * width, (t + 1) * width), pos = state.pos + t;
      for (let i = 0; i < half; i++) { const f = Math.fround(this.invFreq[i] * pos); cos[i] = Math.cos(f); sin[i] = Math.sin(f); }
      for (let head = 0; head < NH; head++) {
        const qo = (t * NH + head) * hd;
        q.set(row.subarray(head * 2 * hd, head * 2 * hd + hd), qo);
        gate.set(row.subarray(head * 2 * hd + hd, (head + 1) * 2 * hd), qo);
      }
      this.rmsnorm(q.subarray(t * NH * hd, (t + 1) * NH * hd), NH, hd, L.qNorm, q.subarray(t * NH * hd, (t + 1) * NH * hd));
      this.rmsnorm(row.subarray(NH * 2 * hd, NH * 2 * hd + KV * hd), KV, hd, L.kNorm, kTmp);
      for (let head = 0; head < NH; head++) rope(q, (t * NH + head) * hd, cos, sin, half);
      for (let kh = 0; kh < KV; kh++) rope(kTmp, kh * hd, cos, sin, half);
      for (let i = 0; i < KV * hd; i++) {
        kNew[t * KV * hd + i] = float32ToFloat16(kTmp[i]);
        vNew[t * KV * hd + i] = float32ToFloat16(row[NH * 2 * hd + KV * hd + i]);
      }
    }
    const kvw = KV * hd;
    const K = append16(state.k[l], kNew), V = append16(state.v[l], vNew);
    const out = new Float32Array(n * NH * hd);
    const scale = 1 / Math.sqrt(hd), rep = NH / KV;
    const scores = new Float64Array(state.pos + n), acc = new Float64Array(hd);
    for (let t = 0; t < n; t++) {
      const upto = state.pos + t + 1;
      for (let head = 0; head < NH; head++) {
        const kh = Math.floor(head / rep), qo = (t * NH + head) * hd;
        let max = -Infinity;
        for (let j = 0; j < upto; j++) {
          let s = 0;
          const ko = j * kvw + kh * hd;
          for (let d = 0; d < hd; d++) s += q[qo + d] * F16[K[ko + d]];
          s = Math.fround(Math.fround(s) * scale);
          scores[j] = s; if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j < upto; j++) { scores[j] = Math.exp(scores[j] - max); sum += scores[j]; }
        acc.fill(0);
        for (let j = 0; j < upto; j++) {
          const w = Math.fround(scores[j] / sum), vo = j * kvw + kh * hd;
          for (let d = 0; d < hd; d++) acc[d] += w * F16[V[vo + d]];
        }
        for (let d = 0; d < hd; d++) out[qo + d] = Math.fround(acc[d]) * Math.fround(sigmoid(gate[qo + d]));
      }
    }
    return this.linear.run(`L${l}.out`, out, n);
  }
}

function shiftConv(conv, mixed, convDim, K) {
  if (K > 2) conv.copyWithin(0, convDim);
  conv.set(mixed, (K - 2) * convDim);
}

function l2norm(x, dim, scale) {
  for (let o = 0; o < x.length; o += dim) {
    let s = 0;
    for (let i = 0; i < dim; i++) s += x[o + i] * x[o + i];
    const inv = Math.fround(1 / Math.sqrt(Math.fround(Math.fround(s) + 1e-6)));
    for (let i = 0; i < dim; i++) x[o + i] = Math.fround(x[o + i] * inv) * scale;
  }
}

function rope(x, o, cos, sin, half) {
  for (let i = 0; i < half; i++) {
    const a = x[o + i], b = x[o + half + i];
    x[o + i] = Math.fround(a * cos[i]) - Math.fround(b * sin[i]);
    x[o + half + i] = Math.fround(b * cos[i]) + Math.fround(a * sin[i]);
  }
}

/** Appends to a growable F16 cache ({ data, len }, capacity doubling); returns the backing array. */
function append16(cache, values) {
  if (cache.len + values.length > cache.data.length) {
    const grown = new Uint16Array(Math.max(2 * cache.data.length, cache.len + values.length));
    grown.set(cache.data.subarray(0, cache.len));
    cache.data = grown;
  }
  cache.data.set(values, cache.len);
  cache.len += values.length;
  return cache.data;
}
