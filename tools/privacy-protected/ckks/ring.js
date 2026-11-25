// RNS ring arithmetic for moduli below 2^32: modular products exact in float64, and the negacyclic NTT in Lattigo's
// convention (bit-reversed twiddles, roots[bitrev(j)] = psi^j; output index j holds the evaluation at
// psi^(2*bitrev(j)+1)). The WGSL kernels implement the same transforms, so CPU and GPU agree word for word.

export function mulmod(a, b, q) {
  return mulmodInv(a, b, q, 1 / q);
}

// reduce x (0 <= x < 2^52) modulo q with a precomputed 1 / q: the float quotient is off by at most one.
function red(x, q, iq) {
  let r = x - Math.floor(x * iq) * q;
  if (r < 0) r += q;
  else if (r >= q) r -= q;
  return r;
}

// a * b mod q for 0 <= a, b < q < 2^32, exact in float64 (b split into 16-bit halves), iq = 1 / q.
export function mulmodInv(a, b, q, iq) {
  const bh = Math.floor(b / 65536);
  const bl = b - bh * 65536;
  return red(red(a * bh, q, iq) * 65536 + a * bl, q, iq);
}

export function addmod(a, b, q) {
  const s = a + b;
  return s >= q ? s - q : s;
}

export function submod(a, b, q) {
  const s = a - b;
  return s < 0 ? s + q : s;
}

export function powmod(x, e, q) {
  let r = 1;
  let base = x % q;
  let k = BigInt(e);
  while (k > 0n) {
    if (k & 1n) r = mulmod(r, base, q);
    base = mulmod(base, base, q);
    k >>= 1n;
  }
  return r;
}

export function invmod(a, q) {
  return powmod(a, q - 2, q);
}

export function bigMod(x, q) {
  const r = x % BigInt(q);
  return Number(r < 0n ? r + BigInt(q) : r);
}

export function bitReverse(x, bits) {
  let r = 0;
  for (let i = 0; i < bits; i++) r = (r << 1) | ((x >>> i) & 1);
  return r >>> 0;
}

// Deterministic Miller-Rabin for n < 2^64 (bases cover all 64-bit integers).
export function isPrime(n) {
  const N = BigInt(n);
  if (N < 2n) return false;
  for (const p of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    if (N === p) return true;
    if (N % p === 0n) return false;
  }
  let d = N - 1n;
  let s = 0;
  while ((d & 1n) === 0n) {
    d >>= 1n;
    s++;
  }
  const pw = (b, e) => {
    let r = 1n;
    b %= N;
    while (e > 0n) {
      if (e & 1n) r = (r * b) % N;
      b = (b * b) % N;
      e >>= 1n;
    }
    return r;
  };
  outer: for (const a of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    let x = pw(a, d);
    if (x === 1n || x === N - 1n) continue;
    for (let i = 1; i < s; i++) {
      x = (x * x) % N;
      if (x === N - 1n) continue outer;
    }
    return false;
  }
  return true;
}

// One modulus with its NTT tables: q, psi (a primitive 2N-th root), twiddles split into 16-bit halves for mulmod.
export class Modulus {
  constructor(q, psi, logN) {
    this.q = q;
    this.logN = logN;
    this.n = 1 << logN;
    const n = this.n;
    if (powmod(psi, n, q) !== q - 1) throw new Error(`psi ${psi} is not a primitive 2N-th root modulo ${q}`);
    this.psi = psi;
    const psiInv = invmod(psi, q);
    this.fwd = new Float64Array(n);
    this.inv = new Float64Array(n);
    let w = 1;
    let wi = 1;
    for (let j = 0; j < n; j++) {
      const r = bitReverse(j, logN);
      this.fwd[r] = w;
      this.inv[r] = wi;
      w = mulmod(w, psi, q);
      wi = mulmod(wi, psiInv, q);
    }
    this.nInv = invmod(n, q);
  }

  // In-place forward NTT of a (length n' <= N, a power of two, using the first n' twiddles: Lattigo's sparse NTT).
  ntt(a, len = this.n) {
    const q = this.q;
    const iq = 1 / q;
    const tw = this.fwd;
    let t = len;
    for (let m = 1; m < len; m <<= 1) {
      t >>= 1;
      for (let i = 0; i < m; i++) {
        const w = tw[m + i];
        const wh = Math.floor(w / 65536);
        const wl = w - wh * 65536;
        const j1 = 2 * i * t;
        for (let j = j1; j < j1 + t; j++) {
          const x = a[j + t];
          let h = x * wh;
          h -= Math.floor(h * iq) * q;
          if (h < 0) h += q;
          else if (h >= q) h -= q;
          let v = h * 65536 + x * wl;
          v -= Math.floor(v * iq) * q;
          if (v < 0) v += q;
          else if (v >= q) v -= q;
          const u = a[j];
          const s = u + v;
          a[j] = s >= q ? s - q : s;
          const d = u - v;
          a[j + t] = d < 0 ? d + q : d;
        }
      }
    }
    return a;
  }

  intt(a) {
    const q = this.q;
    const iq = 1 / q;
    const n = this.n;
    const tw = this.inv;
    let t = 1;
    for (let m = n >> 1; m >= 1; m >>= 1) {
      let j1 = 0;
      for (let i = 0; i < m; i++) {
        const w = tw[m + i];
        const wh = Math.floor(w / 65536);
        const wl = w - wh * 65536;
        for (let j = j1; j < j1 + t; j++) {
          const u = a[j];
          const v = a[j + t];
          const s = u + v;
          a[j] = s >= q ? s - q : s;
          let d = u - v;
          if (d < 0) d += q;
          let h = d * wh;
          h -= Math.floor(h * iq) * q;
          if (h < 0) h += q;
          else if (h >= q) h -= q;
          let r = h * 65536 + d * wl;
          r -= Math.floor(r * iq) * q;
          if (r < 0) r += q;
          else if (r >= q) r -= q;
          a[j + t] = r;
        }
        j1 += 2 * t;
      }
      t <<= 1;
    }
    const ni = this.nInv;
    const nh = Math.floor(ni / 65536);
    const nl = ni - nh * 65536;
    for (let j = 0; j < n; j++) a[j] = red(red(a[j] * nh, q, iq) * 65536 + a[j] * nl, q, iq);
    return a;
  }
}

// Lattigo's AutomorphismNTTIndex: out[j] = in[index[j]] applies X -> X^galEl in the NTT domain.
export function automorphismIndex(logN, galEl) {
  const n = 1 << logN;
  const mask = BigInt(2 * n - 1);
  const g = BigInt(galEl);
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const t1 = BigInt(2 * bitReverse(i, logN) + 1);
    const t2 = Number((((g * t1) & mask) - 1n) >> 1n);
    idx[i] = bitReverse(t2, logN);
  }
  return idx;
}

export function galoisElement(k, logN) {
  const m = 1n << BigInt(logN + 1);
  let r = 1n;
  for (let i = 0; i < k; i++) r = (r * 5n) % m;
  return Number(r);
}

export function modInvGaloisElement(g, logN) {
  // g is odd: its inverse modulo 2N.
  const M = 1n << BigInt(logN + 1);
  let [r0, r1, s0, s1] = [BigInt(g) % M, M, 1n, 0n];
  while (r1 !== 0n) {
    const t = r0 / r1;
    [r0, r1] = [r1, r0 - t * r1];
    [s0, s1] = [s1, s0 - t * s1];
  }
  return Number(((s0 % M) + M) % M);
}
