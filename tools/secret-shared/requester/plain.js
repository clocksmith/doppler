/**
 * Plaintext stand-in for the share operators, for the controls and the layer check only: each group's weights are
 * dequantized (Q4_K as ggml does: d * sc * q - dmin * m) and multiplied in float64. It is never used on the
 * secret-shared path, and it is the only place the requester code ever touches projection weights.
 */
import os from 'node:os';
import { Worker, isMainThread, parentPort } from 'node:worker_threads';
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

/** The projection groups of every layer (and the head), as the secret-shared plan names them, from a manifest alone. */
export function planFromManifest(manifest) {
  const groups = [], P = 'model.language_model.';
  const types = manifest.inference.layerPattern.layerTypes.slice(0, manifest.architecture.numLayers);
  const t = name => ({ name, shape: manifest.tensors[name].shape });
  types.forEach((kind, l) => {
    const L = `${P}layers.${l}.`;
    if (kind === 'linear_attention') {
      const a = `${L}linear_attn.`;
      groups.push({ id: `L${l}.in`, tensors: ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a'].map(n => t(`${a}${n}.weight`)) });
      groups.push({ id: `L${l}.out`, tensors: [t(`${a}out_proj.weight`)] });
    } else {
      const a = `${L}self_attn.`;
      groups.push({ id: `L${l}.in`, tensors: ['q_proj', 'k_proj', 'v_proj'].map(n => t(`${a}${n}.weight`)) });
      groups.push({ id: `L${l}.out`, tensors: [t(`${a}o_proj.weight`)] });
    }
    groups.push({ id: `L${l}.up`, tensors: [t(`${L}mlp.gate_proj.weight`), t(`${L}mlp.up_proj.weight`)] });
    groups.push({ id: `L${l}.down`, tensors: [t(`${L}mlp.down_proj.weight`)] });
  });
  groups.push({ id: 'head', tensors: [t(`${P}lm_head.weight`)] });
  for (const g of groups) g.in = g.tensors[0].shape[1];
  return { stages: [{ index: 0, groups }] };
}

/** float64 dot products of weight rows [start, end) with n columns. */
function rowsDot(w, cols, x, n, start, end, y, stride, offset) {
  for (let r = start; r < end; r++) {
    const o = r * cols;
    for (let c = 0; c < n; c++) {
      const xo = c * cols;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
      for (; k + 3 < cols; k += 4) { s0 += w[o + k] * x[xo + k]; s1 += w[o + k + 1] * x[xo + k + 1]; s2 += w[o + k + 2] * x[xo + k + 2]; s3 += w[o + k + 3] * x[xo + k + 3]; }
      for (; k < cols; k++) s0 += w[o + k] * x[xo + k];
      y[c * stride + offset + r] = s0 + s1 + s2 + s3;
    }
  }
}

export class PlainLinear {
  constructor(plan, source, { threads = 1 } = {}) {
    this.groups = new Map();
    for (const stage of plan.stages) for (const group of stage.groups) this.groups.set(group.id, group);
    this.source = source;
    this.cache = new Map();
    this.threads = threads > 1 ? threads : 1;
    if (this.threads > 1) {
      this.workers = Array.from({ length: Math.min(this.threads, os.availableParallelism()) }, () => new Worker(new URL(import.meta.url)));
      this.waiting = new Map(); this.next = 1;
      for (const w of this.workers) {
        w.on('message', ({ id, error }) => { const p = this.waiting.get(id); this.waiting.delete(id); error ? p.reject(Error(error)) : p.resolve(); });
        w.on('error', error => { for (const p of this.waiting.values()) p.reject(error); this.waiting.clear(); });
        w.unref();
      }
    }
  }

  close() { for (const w of this.workers ?? []) w.terminate(); }

  weights(gid) {
    if (!this.cache.has(gid)) {
      const group = this.groups.get(gid);
      this.cache.set(gid, group.tensors.map(t => {
        const w = dequantize(this.source.raw(t.name));
        if (!this.workers) return { rows: t.shape[0], cols: t.shape[1], w };
        const shared = new Float32Array(new SharedArrayBuffer(w.byteLength)); shared.set(w);
        return { rows: t.shape[0], cols: t.shape[1], w: shared };
      }));
    }
    return this.cache.get(gid);
  }

  out(gid) { return this.groups.get(gid).tensors.reduce((s, t) => s + t.shape[0], 0); }

  /** float64 W x for n columns, returned as float32 [n * out] (float64 when `exact`). */
  async run(gid, x, n, exact = false) {
    const parts = this.weights(gid), out = this.out(gid), cols = parts[0].cols;
    if (this.workers && !exact) {
      const xs = new Float32Array(new SharedArrayBuffer(4 * x.length)); xs.set(x);
      const y = new Float32Array(new SharedArrayBuffer(4 * n * out)), jobs = [];
      let off = 0;
      for (const { rows, w } of parts) {
        const chunk = Math.ceil(rows / this.workers.length);
        for (let i = 0; i * chunk < rows; i++) {
          const id = this.next++;
          jobs.push(new Promise((resolve, reject) => this.waiting.set(id, { resolve, reject })));
          this.workers[i].ref();
          this.workers[i].postMessage({ id, w, cols, x: xs, n, start: i * chunk, end: Math.min(rows, (i + 1) * chunk), y, stride: out, offset: off });
        }
        off += rows;
      }
      try { await Promise.all(jobs); } finally { for (const w of this.workers) w.unref(); }
      return y;
    }
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

if (!isMainThread) {
  parentPort.on('message', ({ id, w, cols, x, n, start, end, y, stride, offset }) => {
    try { rowsDot(w, cols, x, n, start, end, y, stride, offset); parentPort.postMessage({ id }); }
    catch (error) { parentPort.postMessage({ id, error: String(error?.message ?? error) }); }
  });
}
