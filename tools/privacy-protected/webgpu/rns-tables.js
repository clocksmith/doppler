// Host-side number theory for the WebGPU RNS-CKKS kernels: Montgomery constants (R = 2^32), Lattigo's twiddle
// order, basis-conversion tables, rescale and ModDown constants, and automorphism indices. Every table is public:
// it depends only on the pinned moduli and the primitive roots Lattigo chose (carried in the compiled program).

const TWO32 = 4294967296;

// mulmod: a * b mod q for 0 <= a, b < q < 2^32, exact in float64 (products of a 32-bit and a 16-bit value).
export function mulmod(a, b, q) {
  const bh = Math.floor(b / 65536);
  const bl = b - bh * 65536;
  const r = (a * bh) % q;
  return (r * 65536 + a * bl) % q;
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

// -q^-1 mod 2^32 by Newton iteration.
export function negInv32(q) {
  let inv = q | 0;
  for (let i = 0; i < 5; i++) inv = Math.imul(inv, 2 - Math.imul(q | 0, inv));
  const out = (-inv) >>> 0;
  if ((Math.imul(q | 0, out) >>> 0) !== 0xffffffff) throw new Error(`no inverse of ${q} mod 2^32`);
  return out;
}

export function toMont(x, q) {
  return mulmod(x, TWO32 % q, q);
}

function bigToMont(x, q) {
  return Number(((x % q) << 32n) % q);
}

function modInvBig(a, m) {
  let [r0, r1, s0, s1] = [((a % m) + m) % m, m, 1n, 0n];
  while (r1 !== 0n) {
    const t = r0 / r1;
    [r0, r1] = [r1, r0 - t * r1];
    [s0, s1] = [s1, s0 - t * s1];
  }
  if (r0 !== 1n) throw new Error('not invertible');
  return ((s0 % m) + m) % m;
}

export function bitReverse(x, bits) {
  let r = 0;
  for (let i = 0; i < bits; i++) {
    r = (r << 1) | ((x >>> i) & 1);
  }
  return r >>> 0;
}

// moduli: Q then P (Numbers). Returns the per-modulus vec4 table (q, -q^-1, 2^64 mod q, N^-1 in Montgomery form).
export function modulusTable(moduli, logN) {
  const n = 1 << logN;
  const out = new Uint32Array(4 * moduli.length);
  moduli.forEach((q, i) => {
    const qb = BigInt(q);
    out[4 * i] = q;
    out[4 * i + 1] = negInv32(q);
    out[4 * i + 2] = Number((1n << 64n) % qb);
    out[4 * i + 3] = bigToMont(modInvBig(BigInt(n), qb), qb);
  });
  return out;
}

// Twiddles in Lattigo's layout: roots[bitrev(j)] = psi^j (forward) and psi^-j (inverse), Montgomery form, N words
// per modulus.
export function twiddleTables(moduli, psis, logN) {
  const n = 1 << logN;
  const fwd = new Uint32Array(n * moduli.length);
  const inv = new Uint32Array(n * moduli.length);
  moduli.forEach((q, i) => {
    const psi = Number(psis[i]);
    if (powmod(psi, n, q) !== q - 1) throw new Error(`psi for modulus ${q} is not a primitive 2N-th root`);
    const psiInv = powmod(psi, 2 * n - 1, q);
    let w = 1;
    let wi = 1;
    for (let j = 0; j < n; j++) {
      const r = bitReverse(j, logN);
      fwd[i * n + r] = toMont(w, q);
      inv[i * n + r] = toMont(wi, q);
      w = mulmod(w, psi, q);
      wi = mulmod(wi, psiInv, q);
    }
  });
  return { fwd, inv };
}

// One basis-conversion table (layout documented in he_basis.wgsl) for source moduli src (global indices) into every
// global modulus.
function conversionTable(moduli, src) {
  const all = moduli.map(BigInt);
  const d = src.map((g) => all[g]);
  const k = d.length;
  const M = d.reduce((x, y) => x * y, 1n);
  const half = M >> 1n;
  const out = new Uint32Array(5 * k + all.length * (k + 2));
  const two96 = 1n << 96n;
  d.forEach((di, i) => {
    const hat = M / di;
    out[i] = bigToMont(modInvBig(hat % di, di), di);
    out[k + i] = Number(half % di);
    const f = two96 / di;
    out[2 * k + 3 * i] = Number(f & 0xffffffffn);
    out[2 * k + 3 * i + 1] = Number((f >> 32n) & 0xffffffffn);
    out[2 * k + 3 * i + 2] = Number(f >> 64n);
  });
  all.forEach((p, g) => {
    const rec = 5 * k + g * (k + 2);
    d.forEach((di, i) => {
      out[rec + i] = bigToMont(M / di, p);
    });
    out[rec + k] = bigToMont((p - (M % p)) % p, p);
    out[rec + k + 1] = Number(half % p);
  });
  return out;
}

// Digits of the hybrid key switch at level l: [start, end) prime ranges of alpha = |P| primes each.
export function digitsAt(level, alpha) {
  const out = [];
  for (let s = 0; s <= level; s += alpha) out.push([s, Math.min(s + alpha, level + 1)]);
  return out;
}

// Every table the key switch and rescale need, packed into one buffer. Returns { words, offsets }.
export function constantTables(qCount, pCount, moduli) {
  const parts = [];
  let size = 0;
  const offsets = { digit: new Map(), p: 0, rescale: 0, pInv: 0 };
  const push = (arr) => {
    const off = size;
    parts.push(arr);
    size += arr.length;
    return off;
  };
  for (let level = 0; level < qCount; level++) {
    for (const [s, e] of digitsAt(level, pCount)) {
      const key = `${s}:${e}`;
      if (e - s >= 2 && !offsets.digit.has(key)) {
        offsets.digit.set(key, push(conversionTable(moduli, range(s, e))));
      }
    }
  }
  offsets.p = push(conversionTable(moduli, range(qCount, qCount + pCount)));
  // rescale at level l: q_l^-1 mod q_i (i < l), Montgomery form, at rescale + l * qCount + i.
  const qb = moduli.map(BigInt);
  const resc = new Uint32Array(qCount * qCount);
  for (let l = 1; l < qCount; l++) {
    for (let i = 0; i < l; i++) resc[l * qCount + i] = bigToMont(modInvBig(qb[l] % qb[i], qb[i]), qb[i]);
  }
  offsets.rescale = push(resc);
  // ModDown: P^-1 mod q_i, Montgomery form.
  const P = qb.slice(qCount).reduce((x, y) => x * y, 1n);
  const pinv = new Uint32Array(qCount);
  for (let i = 0; i < qCount; i++) pinv[i] = bigToMont(modInvBig(P % qb[i], qb[i]), qb[i]);
  offsets.pInv = push(pinv);
  const words = new Uint32Array(size);
  let at = 0;
  for (const p of parts) {
    words.set(p, at);
    at += p.length;
  }
  return { words, offsets };
}

function range(a, b) {
  return Array.from({ length: b - a }, (_, i) => a + i);
}

// Lattigo's AutomorphismNTTIndex: out[j] = in[index[j]].
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

// Convert a compiled program's multiplicative constants to Montgomery form, in place (each op owns its words).
export function prepareProgramBlob(program, blob) {
  const q = program.q.map(Number);
  for (const op of program.ops) {
    if (op.op === 'smul' || op.op === 'smac') {
      for (let i = 0; i <= op.level; i++) blob[op.k + i] = toMont(blob[op.k + i], q[i]);
    } else if (op.op === 'pmul') {
      for (let i = 0; i <= op.level; i++) {
        for (let b = 0; b < 128; b++) blob[op.pt + i * 128 + b] = toMont(blob[op.pt + i * 128 + b], q[i]);
      }
    }
  }
  return blob;
}
