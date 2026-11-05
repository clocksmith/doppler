// Float64 reference of the pinned qwen4_exp decoder composition with
// first-order F32 error propagation: every value carries an absolute bound
// that covers its inputs' bounds plus the rounding of its own F32 operation in
// any summation order. Test-only; never a runtime fallback.

const u = 2 ** -24;
const gamma = (n) => (n * u) / (1 - n * u);
const MIN_NORMAL = 2 ** -126;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Exact F32 input (zero error). */
export const exact = (values) => ({ v: Float64Array.from(values), e: new Float64Array(values.length) });

/** y[t, n] = sum_k x[t, k] * W[n, k] (transposed: PyTorch [out, in]) or sum_k x[t, k] * W[k, n]. */
export function linear(x, weight, rows, inF, outF, transposed = true) {
  const v = new Float64Array(rows * outF);
  const e = new Float64Array(rows * outF);
  const g = gamma(inF);
  for (let t = 0; t < rows; t++) {
    const xb = t * inF;
    for (let n = 0; n < outF; n++) {
      let sum = 0; let abs = 0; let prop = 0;
      for (let k = 0; k < inF; k++) {
        const w = transposed ? weight[n * inF + k] : weight[k * outF + n];
        const a = x.v[xb + k];
        sum += a * w; abs += Math.abs(a * w); prop += Math.abs(w) * x.e[xb + k];
      }
      v[t * outF + n] = sum;
      e[t * outF + n] = g * (abs + prop) + prop;
    }
  }
  return { v, e };
}

/** Grouped RMSNorm with (1 + w) over groups of `width`, weight slice per group index mod `groups`. */
export function groupedNorm(x, weight, rows, groups, width, eps) {
  const v = new Float64Array(rows * groups * width);
  const e = new Float64Array(v.length);
  const rounding = 0.5 * (gamma(width + 1) + 2 * u) + 5 * u;
  for (let g = 0; g < rows * groups; g++) {
    const base = g * width;
    const wb = (g % groups) * width;
    let sq = 0; let rel = 0;
    for (let i = 0; i < width; i++) { sq += x.v[base + i] ** 2; rel += Math.abs(x.v[base + i]) * x.e[base + i]; }
    const r = 1 / Math.sqrt(sq / width + eps);
    for (let i = 0; i < width; i++) {
      const y = x.v[base + i] * r * (1 + weight[wb + i]);
      v[base + i] = y;
      e[base + i] = Math.abs(1 + weight[wb + i]) * r * (x.e[base + i] + Math.abs(x.v[base + i]) * rel / sq) + Math.abs(y) * rounding;
    }
  }
  return { v, e };
}

/** Elementwise activation: 'low_rank_silu' silu(x/S), 'mix_sigmoid' sigmoid(x), 'inject_gate' 2 sigmoid(x/S), 'silu'. */
export function activation(mode, x, streams) {
  const v = new Float64Array(x.v.length);
  const e = new Float64Array(x.v.length);
  for (let i = 0; i < v.length; i++) {
    if (mode === 'low_rank_silu' || mode === 'silu') {
      const divisor = mode === 'silu' ? 1 : streams;
      const s = x.v[i] / divisor;
      v[i] = s * sigmoid(s);
      e[i] = 1.0998 * (x.e[i] / divisor + u * Math.abs(s)) + 8 * u * Math.abs(v[i]) + MIN_NORMAL;
    } else if (mode === 'mix_sigmoid') {
      v[i] = sigmoid(x.v[i]);
      e[i] = 0.25 * x.e[i] + 8 * u * v[i] + MIN_NORMAL;
    } else if (mode === 'inject_gate') {
      const s = x.v[i] / streams;
      v[i] = 2 * sigmoid(s);
      e[i] = 0.5 * (x.e[i] / streams + u * Math.abs(s)) + 8 * u * v[i] + MIN_NORMAL;
    } else {
      throw new Error(`unknown activation ${mode}`);
    }
  }
  return { v, e };
}

/** Elementwise sum a + b with rounding. */
export function add(a, b) {
  const v = new Float64Array(a.v.length);
  const e = new Float64Array(a.v.length);
  for (let i = 0; i < v.length; i++) {
    v[i] = a.v[i] + b.v[i];
    e[i] = a.e[i] + b.e[i] + u * Math.abs(v[i]);
  }
  return { v, e };
}

/** Gated residual: branch input and (optionally) injection gate. */
export function gatedResidual(streams, weights, { rows, streamCount, width, rank, eps }) {
  const hc = streamCount * width;
  const normed = groupedNorm(streams, weights.norm, rows, streamCount, width, eps);
  const lowRank = activation('low_rank_silu', linear(normed, weights.down, rows, hc, rank), streamCount);
  const mix = activation('mix_sigmoid', linear(lowRank, weights.up, rows, rank, hc), streamCount);
  const branch = { v: new Float64Array(rows * width), e: new Float64Array(rows * width) };
  for (let t = 0; t < rows; t++) {
    for (let w = 0; w < width; w++) {
      let sum = 0; let abs = 0; let prop = 0;
      for (let s = 0; s < streamCount; s++) {
        const i = (t * streamCount + s) * width + w;
        sum += mix.v[i] * normed.v[i];
        abs += Math.abs(mix.v[i] * normed.v[i]);
        prop += Math.abs(normed.v[i]) * mix.e[i] + Math.abs(mix.v[i]) * normed.e[i];
      }
      branch.v[t * width + w] = sum / streamCount;
      branch.e[t * width + w] = (gamma(streamCount + 1) * abs + prop) / streamCount + u * Math.abs(sum / streamCount);
    }
  }
  const gate = weights.inject ? activation('inject_gate', linear(normed, weights.inject, rows, hc, streamCount), streamCount) : null;
  return { branch, gate };
}

/** streams + branch * gate[stream] with the product rounded first. */
export function inject(streams, branch, gate, { rows, streamCount, width }) {
  const v = new Float64Array(streams.v.length);
  const e = new Float64Array(streams.v.length);
  for (let t = 0; t < rows; t++) {
    for (let s = 0; s < streamCount; s++) {
      const g = gate.v[t * streamCount + s];
      const ge = gate.e[t * streamCount + s];
      for (let w = 0; w < width; w++) {
        const i = (t * streamCount + s) * width + w;
        const b = branch.v[t * width + w];
        const product = b * g;
        v[i] = streams.v[i] + product;
        e[i] = streams.e[i] + Math.abs(g) * branch.e[t * width + w] + Math.abs(b) * ge + u * Math.abs(product) + u * Math.abs(v[i]);
      }
    }
  }
  return { v, e };
}

/**
 * PLE layer on streams with propagated stream error: streams + (gated + silu(conv(norm_conv(gated)))).
 * embedding: exact f64 rows [rows, embedDim]. Convolution history starts with zeros at row 0.
 */
export function ple(streams, embedding, weights, { rows, streamCount, width, embedDim, kernel, dilation, eps }) {
  const hc = streamCount * width;
  const emb = exact(embedding);
  const key = groupedNorm(linear(emb, weights.key, rows, embedDim, hc), weights.normKey, rows, streamCount, width, eps);
  const value = linear(emb, weights.value, rows, embedDim, width);
  const query = groupedNorm(streams, weights.normQuery, rows, streamCount, width, eps);
  const divisor = Math.fround(Math.sqrt(width));
  const gated = { v: new Float64Array(rows * hc), e: new Float64Array(rows * hc) };
  for (let t = 0; t < rows; t++) {
    for (let s = 0; s < streamCount; s++) {
      let g = 0; let abs = 0; let prop = 0;
      for (let d = 0; d < width; d++) {
        const i = t * hc + s * width + d;
        g += key.v[i] * query.v[i]; abs += Math.abs(key.v[i] * query.v[i]);
        prop += Math.abs(query.v[i]) * key.e[i] + Math.abs(key.v[i]) * query.e[i];
      }
      g /= divisor;
      const eg = (gamma(width + 1) * abs + prop) / divisor + u * Math.abs(g);
      const gp = Math.sign(g) * Math.sqrt(Math.max(Math.abs(g), 1e-6));
      const egp = eg / (2 * Math.sqrt(Math.max(Math.abs(g) - eg, 1e-6))) + 3 * u * Math.abs(gp);
      const sig = sigmoid(gp);
      const es = 0.25 * egp + 8 * u * sig;
      for (let d = 0; d < width; d++) {
        const i = t * hc + s * width + d;
        const vv = value.v[t * width + d];
        gated.v[i] = sig * vv;
        gated.e[i] = Math.abs(vv) * es + sig * value.e[t * width + d] + u * Math.abs(sig * vv);
      }
    }
  }
  const normed = groupedNorm(gated, weights.normConv, rows, streamCount, width, eps);
  const state = (kernel - 1) * dilation;
  const conv = { v: new Float64Array(rows * hc), e: new Float64Array(rows * hc) };
  for (let t = 0; t < rows; t++) {
    for (let c = 0; c < hc; c++) {
      let acc = 0; let abs = 0; let prop = 0;
      for (let k = 0; k < kernel; k++) {
        const src = t - state + k * dilation;
        if (src < 0) continue;
        const x = normed.v[src * hc + c]; const w = weights.conv[c * kernel + k];
        acc += w * x; abs += Math.abs(w * x); prop += Math.abs(w) * normed.e[src * hc + c];
      }
      conv.v[t * hc + c] = acc;
      conv.e[t * hc + c] = gamma(kernel) * (abs + prop) + prop;
    }
  }
  return add(streams, add(gated, activation('silu', conv, 1)));
}

/** Max |actual - reference| / bound over `indices` (all when omitted). */
export function boundRatio(actual, reference, indices = null) {
  let worst = 0;
  const count = indices ? indices.length : reference.v.length;
  for (let j = 0; j < count; j++) {
    const i = indices ? indices[j] : j;
    worst = Math.max(worst, Math.abs(actual[j] - reference.v[i]) / reference.e[i]);
  }
  return worst;
}
