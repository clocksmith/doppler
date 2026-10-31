// Independent test oracle: GGUF negative A (already transformed), tiled K heads,
// epsilon inside the L2 root, and sequential F32 state recurrence. No WGSL helpers.
export const GDN_REFERENCE_TOLERANCE = { output: 1e-4, state: 1e-5 };
const f32 = Math.fround;
const silu = x => x / (1 + Math.exp(-x));
const values = (n, rate, scale) => Float32Array.from({ length: n }, (_, i) => Math.sin((i + 1) * rate) * scale);

export function makeGdnFixture({ T = 9, V = 17, NK = 2, NV = 4, window = 4,
  normMode = 'per_head', scale = 0.2, eps = 1e-6, zeroState = false } = {}) {
  const K = 128, Q = NK * K, D = NV * V, C = 2 * Q + D;
  const fixture = { T, V, NK, NV, K, Q, D, C, window, normMode, eps, rmsEps: 1e-6,
    qkv: values(T * C, 0.037, scale), z: values(T * D, 0.023, 0.8),
    a: values(T * NV, 0.17, 1.2), b: values(T * NV, 0.13, 2),
    bias: values(NV, 0.19, 0.2), aLog: Float32Array.from({ length: NV }, (_, h) => -0.25 - h * 0.07),
    norm: Float32Array.from({ length: normMode === 'per_head' ? D : V }, (_, i) => 0.7 + (i % 13) * 0.04),
    convWeight: values(C * window, 0.011, 0.5), convState: values(C * window, 0.021, scale * 0.25),
    state: values(NV * K * V, 0.007, zeroState ? 0 : 0.005) };
  // Exercise both softplus branches and both stable sigmoid signs.
  fixture.a[0] = -25; fixture.a[1] = 25;
  fixture.b[0] = -21; fixture.b[1] = 21;
  return fixture;
}

export function sliceGdnFixture(f, start, count) {
  return { ...f, T: count,
    qkv: f.qkv.slice(start * f.C, (start + count) * f.C),
    z: f.z.slice(start * f.D, (start + count) * f.D),
    a: f.a.slice(start * f.NV, (start + count) * f.NV),
    b: f.b.slice(start * f.NV, (start + count) * f.NV) };
}

export function gdnReference(f) {
  const { T, K, V, NK, NV, Q, D, C, window } = f;
  const convState = f.convState.slice(), state = f.state.slice();
  const output = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const conv = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      const base = c * window;
      for (let w = 0; w + 1 < window; w++) convState[base + w] = convState[base + w + 1];
      convState[base + window - 1] = f.qkv[t * C + c];
      let mixed = 0;
      for (let w = 0; w < window; w++) mixed = f32(mixed + convState[base + w] * f.convWeight[base + w]);
      conv[c] = silu(mixed);
    }
    for (let h = 0; h < NV; h++) {
      const kh = h % NK, q = conv.subarray(kh * K, (kh + 1) * K), k = conv.subarray(Q + kh * K, Q + (kh + 1) * K);
      const qscale = 1 / Math.sqrt(K) / Math.sqrt(q.reduce((sum, x) => sum + x * x, 0) + f.eps);
      const kscale = 1 / Math.sqrt(k.reduce((sum, x) => sum + x * x, 0) + f.eps);
      const x = f32(f.a[t * NV + h] + f.bias[h]);
      const decay = f32(Math.exp(f32(f.aLog[h] * Math.log1p(Math.exp(x)))));
      const beta = f32(1 / (1 + Math.exp(-f.b[t * NV + h])));
      const row = new Float32Array(V);
      for (let v = 0; v < V; v++) {
        let memory = 0;
        for (let d = 0; d < K; d++) {
          const i = (h * K + d) * V + v;
          state[i] = state[i] * decay;
          memory = f32(state[i] * f32(k[d] * kscale) + memory);
        }
        const delta = f32(f32(conv[2 * Q + h * V + v] - memory) * beta);
        let value = 0;
        for (let d = 0; d < K; d++) {
          const i = (h * K + d) * V + v;
          state[i] = f32(f32(k[d] * kscale) * delta + state[i]);
          value = f32(state[i] * f32(q[d] * qscale) + value);
        }
        row[v] = value;
      }
      const invRms = 1 / Math.sqrt(row.reduce((sum, x) => sum + x * x, 0) / V + f.rmsEps);
      for (let v = 0; v < V; v++) {
        const norm = f.norm[f.normMode === 'per_head' ? h * V + v : v];
        output[t * D + h * V + v] = row[v] * invRms * norm * silu(f.z[t * D + h * V + v]);
      }
    }
  }
  return { output, state, convState };
}

export function packGdnInputs(f, abPacked, qkvzPacked) {
  const qkv = new Float32Array(f.T * (f.C + (qkvzPacked ? f.D : 0)));
  for (let t = 0; t < f.T; t++) {
    const base = t * (f.C + (qkvzPacked ? f.D : 0));
    qkv.set(f.qkv.subarray(t * f.C, (t + 1) * f.C), base);
    if (qkvzPacked) qkv.set(f.z.subarray(t * f.D, (t + 1) * f.D), base + f.C);
  }
  const bOffset = abPacked ? f.a.length + 7 : 0;
  const a = abPacked ? new Float32Array(bOffset + f.b.length).fill(37) : f.a;
  if (abPacked) { a.set(f.a); a.set(f.b, bOffset); }
  return { qkv, z: qkvzPacked ? qkv : f.z, a, b: abPacked ? a : f.b, bOffset };
}
