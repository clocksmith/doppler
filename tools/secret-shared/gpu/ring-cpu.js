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

const TWO32 = 4294967296, TWO16 = 65536;

function f16(bits) {
  const s = bits & 0x8000 ? -1 : 1, e = (bits >> 10) & 31, f = bits & 1023;
  return e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
}
/** round half away from zero (Go's math.Round), for the fixed-point scales */
const roundAway = v => (v < 0 ? -Math.round(-v) : Math.round(v));

/** acc[0..1] += sign * (A * S) mod 2^64 (unnormalized doubles), with 0 <= A < 2^32 and S = slo + 2^32 shi, slo, shi < 2^53. */
function mulAcc(acc, A, slo, shi, sign) {
  const s0 = slo % TWO32;
  const s1 = (shi + (slo - s0) / TWO32) % TWO32;
  const a0 = A % TWO16, a1 = (A - a0) / TWO16;
  const p0 = a0 * s0, p1 = a1 * s0;
  const p1lo = p1 % TWO16;
  const low = p0 + p1lo * TWO16;
  const lo = low % TWO32;
  const hi = (low - lo) / TWO32 + (p1 - p1lo) / TWO16 + (Math.imul(A, s1) >>> 0);
  acc[0] += sign * lo;
  acc[1] += sign * hi;
}

function normalize(acc, out, index) {
  let lo = acc[0] % TWO32;
  if (lo < 0) lo += TWO32;
  const carry = (acc[0] - lo) / TWO32;
  let hi = (acc[1] + carry) % TWO32;
  if (hi < 0) hi += TWO32;
  out[2 * index] = lo;
  out[2 * index + 1] = hi;
}

/** Computes rows [start, end) of one Q4_K tensor for n columns into y (u32 words, [n][stride], rows at rowOffset). */
export function q4kRows(q4k, cols, x, n, start, end, fw, y, stride, rowOffset) {
  const blocks = cols / 256, scale = 2 ** fw, acc = new Float64Array(2);
  const q = new Uint8Array(64);
  const view = new DataView(q4k.buffer, q4k.byteOffset, q4k.byteLength);
  // Per column and sub-block, T = sum of the 32 inputs (lo words, hi words), shared by every row.
  const per = cols / 32, T = new Float64Array(n * per * 2);
  for (let c = 0; c < n; c++) for (let s = 0; s < per; s++) {
    let tl = 0, th = 0;
    const o = 2 * (c * cols + 32 * s);
    for (let l = 0; l < 32; l++) { tl += x[o + 2 * l]; th += x[o + 2 * l + 1]; }
    T[2 * (c * per + s)] = tl; T[2 * (c * per + s) + 1] = th;
  }
  const accs = new Float64Array(2 * n);
  for (let r = start; r < end; r++) {
    accs.fill(0);
    for (let b = 0; b < blocks; b++) {
      const o = (r * blocks + b) * 144;
      const d = f16(view.getUint16(o, true)), dmin = f16(view.getUint16(o + 2, true));
      for (let j = 0; j < 4; j++) {
        // Two sub-blocks share 32 bytes of codes: low nibbles (sub-block 2j), high nibbles (2j + 1).
        for (let l = 0; l < 32; l++) { const v = q4k[o + 16 + 32 * j + l]; q[l] = v & 15; q[32 + l] = v >> 4; }
        for (let half = 0; half < 2; half++) {
          const sb = 2 * j + half, sc8 = o + 4;
          const sc = sb < 4 ? q4k[sc8 + sb] & 63 : (q4k[sc8 + sb + 4] & 0xf) | ((q4k[sc8 + sb - 4] >> 6) << 4);
          const m = sb < 4 ? q4k[sc8 + sb + 4] & 63 : (q4k[sc8 + sb + 4] >> 4) | ((q4k[sc8 + sb] >> 6) << 4);
          const A = roundAway(d * sc * scale), B = roundAway(dmin * m * scale);
          if (!(A >= 0 && A < TWO32 && B >= 0 && B < TWO32)) throw Error('Q4_K scale outside the u32 fixed-point range');
          const qo = 32 * half, sub = b * 8 + sb;
          for (let c = 0; c < n; c++) {
            const xo = 2 * (c * cols + b * 256 + 32 * sb);
            let slo = 0, shi = 0;
            for (let l = 0; l < 32; l++) { const w = q[qo + l]; slo += w * x[xo + 2 * l]; shi += w * x[xo + 2 * l + 1]; }
            acc[0] = accs[2 * c]; acc[1] = accs[2 * c + 1];
            mulAcc(acc, A, slo, shi, 1);
            mulAcc(acc, B, T[2 * (c * per + sub)], T[2 * (c * per + sub) + 1], -1);
            accs[2 * c] = acc[0]; accs[2 * c + 1] = acc[1];
          }
        }
      }
      // Keep the unnormalized sums well inside 2^53: fold the low words into the high words every 8 blocks.
      if ((b & 7) === 7) for (let c = 0; c < n; c++) {
        const lo = accs[2 * c] % TWO32;
        accs[2 * c + 1] = (accs[2 * c + 1] + (accs[2 * c] - lo) / TWO32) % TWO32;
        accs[2 * c] = lo;
      }
    }
    for (let c = 0; c < n; c++) { acc[0] = accs[2 * c]; acc[1] = accs[2 * c + 1]; normalize(acc, y, c * stride + rowOffset + r); }
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
      if ((k & 255) === 255) { const lo = acc[0] % TWO32; acc[1] = (acc[1] + (acc[0] - lo) / TWO32) % TWO32; acc[0] = lo; }
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
    tensor(rows, cols, q4kBytes) { return { kind: 'q4k', rows, cols, sab: shared(q4kBytes) }; },
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
          workers[i].postMessage({ id, kind: t.kind, sab: t.sab, cols: t.cols, xs, n, start: i * chunk, end: Math.min(t.rows, (i + 1) * chunk), ys, stride: out, rowOffset: offset });
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
  parentPort.on('message', ({ id, kind, sab, cols, xs, n, start, end, ys, stride, rowOffset }) => {
    try {
      const x = new Uint32Array(xs), y = new Uint32Array(ys);
      if (kind === 'q4k') q4kRows(new Uint8Array(sab), cols, x, n, start, end, workerData.fw, y, stride, rowOffset);
      else denseRows(new Float64Array(sab), cols, x, n, start, end, y, stride, rowOffset);
      parentPort.postMessage({ id });
    } catch (error) { parentPort.postMessage({ id, error: String(error?.message ?? error) }); }
  });
}
