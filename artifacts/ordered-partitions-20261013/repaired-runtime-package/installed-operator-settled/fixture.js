let seed = 624901;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; }
const pairValues = (count, scale) => Array.from({ length: count }, () => [Math.fround((random() - .5) * scale), Math.fround((random() - .5) * scale * 2 ** -26)]).flat();
const scalarValues = (count, scale) => Array.from({ length: count }, () => Math.fround((random() - .5) * scale));
const cases = [false, true].map(packed => {
  const tokens = 3, nk = 1, nv = 2, kd = 128, vd = 128, kernel = 4, convDim = (2 * nk * kd + nv * vd), valueDim = nv * vd;
  return { tokens, nk, nv, kd, vd, kernel, convDim, valueDim, packed, eps: 1e-6, qeps: 1e-6, offset: packed,
    qkv: pairValues(tokens * (convDim + (packed ? valueDim : 0)), .8), z: pairValues(tokens * valueDim, 1.4),
    a: pairValues(tokens * nv, 1.2), b: pairValues(tokens * nv, 1.2),
    weight: scalarValues(convDim * kernel, .8), bias: scalarValues(nv, .3), log: scalarValues(nv, .8),
    norm: scalarValues(vd, .5), convState: pairValues(convDim * kernel, .2), state: pairValues(nv * kd * vd, .01) };
});
function reference(f) {
  const unpack = values => Array.from({ length: values.length / 2 }, (_, i) => values[i * 2] + values[i * 2 + 1]);
  const qkv = unpack(f.qkv), z = unpack(f.z), a = unpack(f.a), b = unpack(f.b), convState = unpack(f.convState), state = unpack(f.state);
  const conv = new Array(f.tokens * f.convDim), output = [];
  const sigmoid = value => 1 / (1 + Math.exp(-value));
  for (let t = 0; t < f.tokens; t++) {
    for (let c = 0; c < f.convDim; c++) {
      const base = c * f.kernel;
      for (let k = 0; k < f.kernel - 1; k++) convState[base + k] = convState[base + k + 1];
      convState[base + f.kernel - 1] = qkv[t * (f.convDim + (f.packed ? f.valueDim : 0)) + c];
      let sum = 0; for (let k = 0; k < f.kernel; k++) sum += convState[base + k] * f.weight[base + k];
      conv[t * f.convDim + c] = sum * sigmoid(sum);
    }
    for (let h = 0; h < f.nv; h++) {
      const row = t * f.convDim, kh = Math.floor(h / (f.nv / f.nk)), qb = kh * f.kd, kb = f.nk * f.kd + qb;
      const q = conv.slice(row + qb, row + qb + f.kd), k = conv.slice(row + kb, row + kb + f.kd);
      const qi = 1 / Math.sqrt(q.reduce((s, v) => s + v * v, 0) + f.qeps) / Math.sqrt(f.kd);
      const ki = 1 / Math.sqrt(k.reduce((s, v) => s + v * v, 0) + f.qeps);
      const beta = sigmoid(b[t * f.nv + h]);
      const decay = Math.exp(-Math.exp(f.log[h]) * Math.log1p(Math.exp(a[t * f.nv + h] + f.bias[h])));
      const raw = [];
      for (let v = 0; v < f.vd; v++) {
        let pred = 0;
        for (let d = 0; d < f.kd; d++) { const index = (h * f.kd + d) * f.vd + v; state[index] *= decay; pred += state[index] * k[d] * ki; }
        const delta = (conv[row + 2 * f.nk * f.kd + h * f.vd + v] - pred) * beta;
        let value = 0;
        for (let d = 0; d < f.kd; d++) { const index = (h * f.kd + d) * f.vd + v; state[index] += k[d] * ki * delta; value += state[index] * q[d] * qi; }
        raw.push(value);
      }
      const inv = 1 / Math.sqrt(raw.reduce((s, v) => s + v * v, 0) / f.vd + f.eps);
      for (let v = 0; v < f.vd; v++) {
        const gate = f.packed ? qkv[t * (f.convDim + f.valueDim) + f.convDim + h * f.vd + v] : z[t * f.valueDim + h * f.vd + v];
        output.push(raw[v] * inv * (f.norm[v] + Number(f.offset)) * gate * sigmoid(gate));
      }
    }
  }
  return { conv, convState, state, output };
}

export { cases, reference };
