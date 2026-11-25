/**
 * Plaintext stand-in for the share operators, for the controls and the layer check only: each group's weights are
 * dequantized (Q4_K as ggml does: d * sc * q - dmin * m) and multiplied in float64. It is never used on the
 * secret-shared path, and it is the only place the requester code ever touches projection weights.
 */
import { float16ToFloat32 } from '../../../src/formats/float16.js';

export function scaleMin(s, j) {
  if (j < 4) return [s[j] & 63, s[j + 4] & 63];
  return [(s[j + 4] & 0xf) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)];
}

export function dequantize({ dtype, shape, bytes }) {
  const [rows, cols] = shape, out = new Float32Array(rows * cols);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dtype === 'F16') { for (let i = 0; i < out.length; i++) out[i] = float16ToFloat32(view.getUint16(2 * i, true)); return out; }
  if (dtype === 'F32') { for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(4 * i, true); return out; }
  if (!dtype.startsWith('Q4_K')) throw Error(`dtype ${dtype}`);
  const blocks = (rows * cols) / 256;
  for (let b = 0; b < blocks; b++) {
    const o = b * 144;
    const d = float16ToFloat32(view.getUint16(o, true)), dmin = float16ToFloat32(view.getUint16(o + 2, true));
    const scales = bytes.subarray(o + 4, o + 16), qs = bytes.subarray(o + 16, o + 144);
    for (let j = 0; j < 8; j++) {
      const [sc, m] = scaleMin(scales, j);
      const a = Math.fround(d * sc), c = Math.fround(dmin * m);
      for (let l = 0; l < 32; l++) {
        const q = j % 2 ? qs[32 * (j >> 1) + l] >> 4 : qs[32 * (j >> 1) + l] & 15;
        out[b * 256 + 32 * j + l] = Math.fround(a * q) - c;
      }
    }
  }
  return out;
}

export class PlainLinear {
  constructor(plan, source) {
    this.groups = new Map();
    for (const stage of plan.stages) for (const group of stage.groups) this.groups.set(group.id, group);
    this.source = source;
    this.cache = new Map();
  }

  weights(gid) {
    if (!this.cache.has(gid)) {
      const group = this.groups.get(gid);
      this.cache.set(gid, group.tensors.map(t => ({ rows: t.shape[0], cols: t.shape[1], w: dequantize(this.source.raw(t.name)) })));
    }
    return this.cache.get(gid);
  }

  out(gid) { return this.groups.get(gid).tensors.reduce((s, t) => s + t.shape[0], 0); }

  /** float64 W x for n columns, returned as float32 [n * out] (float64 when `exact`). */
  async run(gid, x, n, exact = false) {
    const parts = this.weights(gid), out = this.out(gid), cols = parts[0].cols;
    const y = exact ? new Float64Array(n * out) : new Float32Array(n * out);
    for (let c = 0; c < n; c++) {
      let off = 0;
      for (const { rows, w } of parts) {
        for (let r = 0; r < rows; r++) {
          let s = 0;
          const o = r * cols, xo = c * cols;
          for (let k = 0; k < cols; k++) s += w[o + k] * x[xo + k];
          y[c * out + off + r] = s;
        }
        off += rows;
      }
    }
    return y;
  }
}
