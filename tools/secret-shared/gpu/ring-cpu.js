/**
 * Exact Z_(2^64) ring matvec on the CPU in plain JS, for operators without a usable GPU. Same algebra as the WGSL
 * kernel (ring-matvec.js) and the Go operator: for a row and a 32-weight Q4_K sub-block with scale A = round(d*sc*2^fw)
 * and offset B = round(dmin*m*2^fw),
 *   contribution = A * S - B * T   (mod 2^64),   S = sum q * x,   T = sum x.
 * Each 64-bit input share is two u32 words (lo, hi). S and T are kept as two exact double sums (lo words and hi
 * words, each < 2^42), and A * S mod 2^64 is built from 16-bit pieces of A so every intermediate stays below 2^53:
 * nothing is approximated. F16/F32 projections (DeltaNet in_proj_a and in_proj_b) use the same product per weight.
 *
 * Rows are split over worker threads that share the weights, the input and the output through SharedArrayBuffers.
 */
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import os from 'node:os';

const TWO32 = 4294967296, TWO16 = 65536, INV32 = 1 / TWO32, INV16 = 1 / TWO16;
// x mod m for integer-valued doubles below 2^53, by floor (Math.floor is an instruction; % on doubles is an fmod call)
const mod32 = v => v - Math.floor(v * INV32) * TWO32;
const mod16 = v => v - Math.floor(v * INV16) * TWO16;

function f16(bits) {
  const s = bits & 0x8000 ? -1 : 1, e = (bits >> 10) & 31, f = bits & 1023;
  return e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
}
/** round half away from zero (Go's math.Round), for the fixed-point scales */
const roundAway = v => (v < 0 ? -Math.round(-v) : Math.round(v));

/** acc[0..1] += sign * (A * S) mod 2^64 (unnormalized doubles), with 0 <= A < 2^32 and S = slo + 2^32 shi, slo, shi < 2^53. */
function mulAcc(acc, A, slo, shi, sign) {
  const s0 = mod32(slo);
  const s1 = mod32(shi + (slo - s0) * INV32);
  const a0 = mod16(A), a1 = (A - a0) * INV16;
  const p0 = a0 * s0, p1 = a1 * s0;
  const p1lo = mod16(p1);
  const low = p0 + p1lo * TWO16;
  const lo = mod32(low);
  const hi = (low - lo) / TWO32 + (p1 - p1lo) / TWO16 + (Math.imul(A, s1) >>> 0);
  acc[0] += sign * lo;
  acc[1] += sign * hi;
}

function normalize(acc, out, index) {
  const lo = mod32(acc[0]);
  const hi = mod32(acc[1] + (acc[0] - lo) * INV32);
  out[2 * index] = lo;
  out[2 * index + 1] = hi;
}

/** The fixed-point scale A = round(d*sc*2^fw) and offset B = round(dmin*m*2^fw) of every 32-weight sub-block, once per
 *  tensor, as 16-bit halves (a0, a1, b0, b1). */
export function q4kScales(q4k, rows, cols, fw) {
  const per = cols / 32, blocks = cols / 256, scale = 2 ** fw, AB = new Uint16Array(rows * per * 4);
  const view = new DataView(q4k.buffer, q4k.byteOffset, q4k.byteLength);
  for (let r = 0; r < rows; r++) for (let b = 0; b < blocks; b++) {
    const o = (r * blocks + b) * 144, d = f16(view.getUint16(o, true)), dmin = f16(view.getUint16(o + 2, true)), sc8 = o + 4;
    for (let sb = 0; sb < 8; sb++) {
      const sc = sb < 4 ? q4k[sc8 + sb] & 63 : (q4k[sc8 + sb + 4] & 0xf) | ((q4k[sc8 + sb - 4] >> 6) << 4);
      const m = sb < 4 ? q4k[sc8 + sb + 4] & 63 : (q4k[sc8 + sb + 4] >> 4) | ((q4k[sc8 + sb] >> 6) << 4);
      const A = roundAway(d * sc * scale), B = roundAway(dmin * m * scale);
      if (!(A >= 0 && A < TWO32 && B >= 0 && B < TWO32)) throw Error('Q4_K scale outside the u32 fixed-point range');
      const i = 4 * (r * per + b * 8 + sb);
      AB[i] = A % TWO16; AB[i + 1] = Math.floor(A / TWO16); AB[i + 2] = B % TWO16; AB[i + 3] = Math.floor(B / TWO16);
    }
  }
  return AB;
}

/**
 * Computes rows [start, end) of one Q4_K tensor for n columns into y (u32 words, [n][stride], rows at rowOffset).
 * Per sub-block and column, S = sum q * x is two exact sums of u32 words (< 2^41 each), reduced to words s0, s1 mod
 * 2^64; with A = a0 + 2^16 a1, the products a0*s0, a1*s0, a0*s1, a1*s1 (each < 2^48) accumulate exactly for 16
 * sub-blocks and fold into the row's 64-bit sum as a0*s0 + 2^16 a1*s0 + 2^32 a0*s1 + 2^48 a1*s1 (mod 2^64). The
 * offset term B * T is the same with T's words precomputed per round. AB comes from q4kScales (or is computed here).
 */
export function q4kRows(q4k, cols, x, n, start, end, fw, y, stride, rowOffset, AB = null) {
  const per = cols / 32, blocks = cols / 256;
  if (!AB) AB = q4kScales(q4k.subarray(start * blocks * 144, end * blocks * 144), end - start, cols, fw), AB.base = start;
  const base = AB.base ?? 0;
  const xd = Float64Array.from(x);                   // the input words as doubles, read 32x per weight row
  const T = new Float64Array(n * per * 2);           // per column and sub-block: sum of the inputs, as words mod 2^64
  for (let c = 0; c < n; c++) for (let s = 0; s < per; s++) {
    let tl = 0, th = 0;
    const o = 2 * (c * cols + 32 * s);
    for (let l = 0; l < 32; l++) { tl += xd[o + 2 * l]; th += xd[o + 2 * l + 1]; }
    const t0 = mod32(tl);
    T[2 * (c * per + s)] = t0; T[2 * (c * per + s) + 1] = mod32(th + (tl - t0) * INV32);
  }
  const q = new Float64Array(64);
  const P = new Float64Array(8 * n);                 // per column: A-term a0s0, a1s0, a0s1, a1s1, then the B-term's
  const W = new Float64Array(2 * n);                 // per column: the row's 64-bit sum as unnormalized (lo, hi) words
  const fold = () => {
    for (let c = 0; c < n; c++) {
      let lo = W[2 * c], hi = W[2 * c + 1];
      for (let t = 0; t < 2; t++) {
        const o = 8 * c + 4 * t, sign = t ? -1 : 1;
        const p00 = P[o], p10 = P[o + 1], q0 = P[o + 2], q1 = P[o + 3];
        const pl = mod32(p00), u0 = mod16(p10);
        lo += sign * (pl + u0 * TWO16);
        hi += sign * ((p00 - pl) * INV32 + (p10 - u0) * INV16 + mod32(q0) + mod16(q1) * TWO16);
        P[o] = P[o + 1] = P[o + 2] = P[o + 3] = 0;
      }
      const l = mod32(lo);
      W[2 * c] = l; W[2 * c + 1] = mod32(hi + (lo - l) * INV32);
    }
  };
  for (let r = start; r < end; r++) {
    W.fill(0);
    for (let b = 0; b < blocks; b++) {
      const o = (r * blocks + b) * 144;
      for (let j = 0; j < 4; j++) {
        // Two sub-blocks share 32 bytes of codes: low nibbles (sub-block 2j), high nibbles (2j + 1).
        for (let l = 0; l < 32; l++) { const v = q4k[o + 16 + 32 * j + l]; q[l] = v & 15; q[32 + l] = v >> 4; }
        for (let half = 0; half < 2; half++) {
          const sb = 2 * j + half, sub = b * 8 + sb, ai = 4 * ((r - base) * per + sub);
          const a0 = AB[ai], a1 = AB[ai + 1], b0 = AB[ai + 2], b1 = AB[ai + 3];
          const qo = 32 * half;
          for (let c = 0; c < n; c++) {
            const xo = 2 * (c * cols + b * 256 + 32 * sb);
            let slo = 0, shi = 0;
            for (let l = 0; l < 32; l++) { const w = q[qo + l]; slo += w * xd[xo + 2 * l]; shi += w * xd[xo + 2 * l + 1]; }
            const s0 = mod32(slo), s1 = mod32(shi + (slo - s0) * INV32);
            const t0 = T[2 * (c * per + sub)], t1 = T[2 * (c * per + sub) + 1], po = 8 * c;
            P[po] += a0 * s0; P[po + 1] += a1 * s0; P[po + 2] += a0 * s1; P[po + 3] += a1 * s1;
            P[po + 4] += b0 * t0; P[po + 5] += b1 * t0; P[po + 6] += b0 * t1; P[po + 7] += b1 * t1;
          }
        }
      }
      if ((b & 1) === 1) fold();                     // 16 sub-blocks: every accumulator stays below 2^52
    }
    fold();
    for (let c = 0; c < n; c++) {
      const lo = mod32(W[2 * c]), hi = mod32(W[2 * c + 1] + (W[2 * c] - lo) * INV32);
      y[2 * (c * stride + rowOffset + r)] = lo; y[2 * (c * stride + rowOffset + r) + 1] = hi;
    }
  }
}

/** Dense fixed-point rows: w are signed integers round(v * 2^fw), |w| < 2^32. */
export function denseRows(w, cols, x, n, start, end, y, stride, rowOffset) {
  const acc = new Float64Array(2);
  for (let r = start; r < end; r++) for (let c = 0; c < n; c++) {
    acc[0] = 0; acc[1] = 0;
    for (let k = 0; k < cols; k++) {
      const v = w[r * cols + k];
      if (v === 0) continue;
      mulAcc(acc, v < 0 ? -v : v, x[2 * (c * cols + k)], x[2 * (c * cols + k) + 1], v < 0 ? -1 : 1);
      if ((k & 255) === 255) { const lo = mod32(acc[0]); acc[1] = mod32(acc[1] + (acc[0] - lo) * INV32); acc[0] = lo; }
    }
    normalize(acc, y, c * stride + rowOffset + r);
  }
}

/** A CPU ring matvec with the same interface as createRingMatVec (ring-matvec.js). */
export function createCpuRingMatVec({ fw, threads = 0 }) {
  const count = threads > 0 ? threads : os.availableParallelism();
  const workers = Array.from({ length: count }, () => new Worker(new URL(import.meta.url), { workerData: { fw } }));
  let nextId = 1;
  const waiting = new Map();
  for (const w of workers) { w.on('message', ({ id, error }) => { const done = waiting.get(id); waiting.delete(id); error ? done.reject(Error(error)) : done.resolve(); }); w.unref(); }
  const shared = bytes => { const sab = new SharedArrayBuffer(bytes.byteLength); new Uint8Array(sab).set(bytes); return sab; };
  return {
    threads: count,
    tensor(rows, cols, q4kBytes) {
      const ab = q4kScales(q4kBytes, rows, cols, fw), abs = new SharedArrayBuffer(ab.byteLength);
      new Uint16Array(abs).set(ab);
      return { kind: 'q4k', rows, cols, sab: shared(q4kBytes), abs };
    },
    dense(rows, cols, values) { const sab = new SharedArrayBuffer(8 * rows * cols); new Float64Array(sab).set(values); return { kind: 'dense', rows, cols, sab }; },
    /** x: Uint32Array [n*cols*2]; returns Uint32Array [n*out*2] for the members in order. */
    async run(tensors, x, n, out) {
      const xs = new SharedArrayBuffer(x.byteLength); new Uint32Array(xs).set(x);
      const ys = new SharedArrayBuffer(n * out * 8);
      const jobs = [];
      let offset = 0;
      for (const t of tensors) {
        const chunk = Math.ceil(t.rows / count);
        for (let i = 0; i < count && i * chunk < t.rows; i++) {
          const id = nextId++;
          jobs.push(new Promise((resolve, reject) => waiting.set(id, { resolve, reject })));
          workers[i].postMessage({ id, kind: t.kind, sab: t.sab, abs: t.abs, cols: t.cols, xs, n, start: i * chunk, end: Math.min(t.rows, (i + 1) * chunk), ys, stride: out, rowOffset: offset });
        }
        offset += t.rows;
      }
      for (const w of workers) w.ref();
      try { await Promise.all(jobs); } finally { for (const w of workers) w.unref(); }
      return new Uint32Array(ys);
    },
    close() { for (const w of workers) w.terminate(); }
  };
}

if (!isMainThread && workerData) {
  parentPort.on('message', ({ id, kind, sab, abs, cols, xs, n, start, end, ys, stride, rowOffset }) => {
    try {
      const x = new Uint32Array(xs), y = new Uint32Array(ys);
      if (kind === 'q4k') q4kRows(new Uint8Array(sab), cols, x, n, start, end, workerData.fw, y, stride, rowOffset, new Uint16Array(abs));
      else denseRows(new Float64Array(sab), cols, x, n, start, end, y, stride, rowOffset);
      parentPort.postMessage({ id });
    } catch (error) { parentPort.postMessage({ id, error: String(error?.message ?? error) }); }
  });
}
