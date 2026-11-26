// CPU execution of the compiled program's RNS operations (rnsOp), word for word as the WebGPU kernels compute them:
// the reference for kernel tests and the second evaluator of the dual check on the development ring. Public
// material only (evaluation keys and ciphertexts).
//
// A ciphertext is { deg, level, body: Uint32Array((deg+1) x (level+1) x N) }, component then limb then coefficient,
// in the NTT domain (the raw-u32 body). Keys are raw-u32 key bodies (canonical residues).

import { ringOf } from './params.js';
import { COMPACT_WORDS } from './encoder.js';
import { automorphismIndex, invmod, mulmodInv } from './ring.js';

function range(a, b) {
  return Array.from({ length: b - a }, (_, i) => a + i);
}

export function digitsAt(level, alpha) {
  const out = [];
  for (let s = 0; s <= level; s += alpha) out.push([s, Math.min(s + alpha, level + 1)]);
  return out;
}

// Basis conversion from source moduli (global indices) into target moduli: the centred lift of he_basis.wgsl, with
// the overflow count v = floor(sum y_i * floor(2^96 / d_i) / 2^96).
class Conversion {
  constructor(moduli, src) {
    const d = src.map((g) => BigInt(moduli[g]));
    const k = d.length;
    const M = d.reduce((x, y) => x * y, 1n);
    this.k = k;
    this.dn = src.map((g) => moduli[g]);
    this.half = M >> 1n;
    this.halfMod = d.map((di) => Number(this.half % di));
    this.hatInv = d.map((di) => Number(modInvBig((M / di) % di, di)));
    this.F = d.map((di) => (1n << 96n) / di);
    this.moduli = moduli;
    this.M = M;
    this.targets = new Map();
  }

  target(g) {
    let t = this.targets.get(g);
    if (!t) {
      const p = BigInt(this.moduli[g]);
      t = {
        hat: this.dn.map((di) => Number((this.M / BigInt(di)) % p)),
        negM: Number((p - (this.M % p)) % p),
        half: Number(this.half % p),
      };
      this.targets.set(g, t);
    }
    return t;
  }

  // src: k rows (coefficient domain); returns y rows and v.
  prepare(rows, n) {
    const k = this.k;
    const y = rows.map(() => new Float64Array(n));
    const v = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < k; i++) {
        const d = this.dn[i];
        let x = rows[i][j] + this.halfMod[i];
        if (x >= d) x -= d;
        const yi = mulmodInv(x, this.hatInv[i], d, 1 / d);
        y[i][j] = yi;
        s += yi / d;
      }
      // floor(sum y_i * floor(2^96/d_i) / 2^96): the float64 sum decides it unless it is within 1e-9 of an integer.
      const f = Math.floor(s);
      if (s - f > 1e-9 && f + 1 - s > 1e-9) {
        v[j] = f;
      } else {
        let acc = 0n;
        for (let i = 0; i < k; i++) acc += BigInt(y[i][j]) * this.F[i];
        v[j] = Number((acc >> 96n) & 0xffffffffn);
      }
    }
    return { y, v };
  }

  // out_j = sum_i y_i * (M/d_i mod p) + v * (-M mod p) - floor(M/2) mod p. The products are summed unreduced in
  // two 16-bit halves of the constants (each partial sum stays below 2^53), then reduced once.
  into(prep, g, n) {
    const t = this.target(g);
    const q = this.moduli[g];
    const iq = 1 / q;
    const k = this.k;
    const hh = t.hat.map((h) => Math.floor(h / 65536));
    const hl = t.hat.map((h, i) => h - hh[i] * 65536);
    const out = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let sh = 0;
      let sl = 0;
      for (let i = 0; i < k; i++) {
        const y = prep.y[i][j];
        sh += y * hh[i];
        sl += y * hl[i];
      }
      let h = sh - Math.floor(sh * iq) * q;
      if (h < 0) h += q;
      else if (h >= q) h -= q;
      let acc = h * 65536 + sl;
      acc -= Math.floor(acc * iq) * q;
      if (acc < 0) acc += q;
      else if (acc >= q) acc -= q;
      acc += mulmodInv(prep.v[j], t.negM, q, iq);
      if (acc >= q) acc -= q;
      acc -= t.half;
      if (acc < 0) acc += q;
      out[j] = acc;
    }
    return out;
  }
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

export class RnsExec {
  constructor(profile) {
    this.p = profile;
    this.n = profile.n;
    const ring = ringOf(profile);
    this.mods = [...ring.Q, ...ring.P];
    this.moduli = this.mods.map((m) => m.q);
    this.keys = { relin: null, galois: new Map() };
    this.conv = new Map();
    this.perm = new Map();
    const P = profile.p.reduce((x, y) => x * BigInt(y), 1n);
    this.pInv = profile.q.map((q) => Number(modInvBig(P % BigInt(q), BigInt(q))));
    this.rescaleInv = new Map();
  }

  loadKey(kind, gal, body) {
    if (kind === 0) this.keys.relin = body;
    else this.keys.galois.set(gal, body);
  }

  conversion(src) {
    const key = src.join(',');
    if (!this.conv.has(key)) this.conv.set(key, new Conversion(this.moduli, src));
    return this.conv.get(key);
  }

  newCt(deg, level) {
    return { deg, level, body: new Uint32Array((deg + 1) * (level + 1) * this.n) };
  }

  // limb view of component c at limb i of a ciphertext of level `level`
  row(ct, c, i) {
    const n = this.n;
    const off = (c * (ct.level + 1) + i) * n;
    return ct.body.subarray(off, off + n);
  }

  // Hybrid key switch of polynomial src (NTT domain, limbs 0..l, an array of rows) with a gadget key body.
  keySwitch(src, l, key) {
    const n = this.n;
    const p = this.p;
    const Q = p.qCount;
    const Pc = p.pCount;
    const QP = Q + Pc;
    const inv = src.map((r, i) => this.mods[i].intt(Float64Array.from(r)));
    const acc = [new Map(), new Map()];
    const limbsUsed = [...range(0, l + 1), ...range(Q, QP)];
    digitsAt(l, Pc).forEach(([s, e], d) => {
      const k = e - s;
      const ext = new Map();
      const targets = [...range(0, s), ...range(e, l + 1), ...range(Q, QP)];
      if (k >= 2) {
        const conv = this.conversion(range(s, e));
        const prep = conv.prepare(inv.slice(s, e), n);
        for (const g of targets) ext.set(g, this.mods[g].ntt(conv.into(prep, g, n)));
      } else {
        const dq = this.moduli[s];
        const x = inv[s];
        for (const g of targets) {
          const q = this.moduli[g];
          const row = new Float64Array(n);
          for (let j = 0; j < n; j++) {
            if (x[j] >= Math.floor(dq / 2)) {
              const r = (dq - x[j]) % q;
              row[j] = r === 0 ? 0 : q - r;
            } else {
              row[j] = x[j] % q;
            }
          }
          ext.set(g, this.mods[g].ntt(row));
        }
      }
      for (let i = s; i < e; i++) ext.set(i, Float64Array.from(src[i]));
      for (const g of limbsUsed) {
        const q = this.moduli[g];
        const iq = 1 / q;
        const xr = ext.get(g);
        for (let c = 0; c < 2; c++) {
          const kOff = ((d * 2 + c) * QP + g) * n;
          let a = acc[c].get(g);
          if (!a) {
            a = new Float64Array(n);
            acc[c].set(g, a);
          }
          for (let j = 0; j < n; j++) {
            const v = a[j] + mulmodInv(xr[j], key[kOff + j], q, iq);
            a[j] = v >= q ? v - q : v;
          }
        }
      }
    });
    const out = [];
    const convP = this.conversion(range(Q, QP));
    for (let c = 0; c < 2; c++) {
      const pRows = range(Q, QP).map((g) => this.mods[g].intt(Float64Array.from(acc[c].get(g))));
      const prep = convP.prepare(pRows, n);
      const rows = [];
      for (let i = 0; i <= l; i++) {
        const q = this.moduli[i];
        const conv = this.mods[i].ntt(convP.into(prep, i, n));
        const a = acc[c].get(i);
        const row = new Float64Array(n);
        const pi = this.pInv[i];
        for (let j = 0; j < n; j++) {
          let t = a[j] - conv[j];
          if (t < 0) t += q;
          row[j] = mulmodInv(t, pi, q, 1 / q);
        }
        rows.push(row);
      }
      out.push(rows);
    }
    return out;
  }

  // Executes one op; a, b are input ciphertexts (read at op.level), blob the program constants (canonical words).
  run(op, a, b, blob) {
    const l = op.level;
    const n = this.n;
    const lim = l + 1;
    const out = this.newCt(op.op === 'relin' || op.op === 'rotate' ? 1 : op.op === 'tensor' ? 2 : op.op === 'smul' || op.op === 'pmul'
      || op.op === 'sadd' || op.op === 'padd' || op.op === 'rescale' ? a.deg : op.deg, op.op === 'rescale' ? l - 1 : l);
    const R = (ct, c, i) => this.row(ct, c, i);
    const W = (c, i) => this.row(out, c, i);
    const each = (fn) => {
      for (let i = 0; i < lim; i++) {
        const q = this.moduli[i];
        fn(i, q, 1 / q);
      }
    };
    switch (op.op) {
      case 'zero':
        return out;
      case 'add':
      case 'sub':
        for (let c = 0; c <= op.deg; c++) {
          each((i, q, iq) => {
            const o = W(c, i);
            const x = c <= a.deg ? R(a, c, i) : null;
            const y = c <= b.deg ? R(b, c, i) : null;
            for (let j = 0; j < n; j++) {
              if (x && y) {
                const v = op.op === 'add' ? x[j] + y[j] : x[j] - y[j];
                o[j] = v >= q ? v - q : v < 0 ? v + q : v;
              } else if (x) o[j] = x[j];
              else if (op.op === 'add') o[j] = y[j];
              else o[j] = y[j] === 0 ? 0 : q - y[j];
            }
          });
        }
        return out;
      case 'smul':
        for (let c = 0; c <= a.deg; c++) {
          each((i, q, iq) => {
            const k = blob[op.k + i];
            const x = R(a, c, i);
            const o = W(c, i);
            for (let j = 0; j < n; j++) o[j] = mulmodInv(x[j], k, q, iq);
          });
        }
        return out;
      case 'sadd':
      case 'padd':
        for (let c = 0; c <= a.deg; c++) {
          each((i, q, iq) => {
            const x = R(a, c, i);
            const o = W(c, i);
            if (c > 0) {
              o.set(x);
              return;
            }
            const block = n / COMPACT_WORDS;
            for (let j = 0; j < n; j++) {
              const k = op.op === 'sadd' ? blob[op.k + i] : blob[op.pt + i * COMPACT_WORDS + Math.floor(j / block)];
              const v = x[j] + k;
              o[j] = v >= q ? v - q : v;
            }
          });
        }
        return out;
      case 'smac':
        for (let c = 0; c <= op.deg; c++) {
          each((i, q, iq) => {
            const acc = R(b, c, i);
            const o = W(c, i);
            if (c > a.deg) {
              o.set(acc);
              return;
            }
            const k = blob[op.k + i];
            const x = R(a, c, i);
            for (let j = 0; j < n; j++) {
              const v = acc[j] + mulmodInv(x[j], k, q, iq);
              o[j] = v >= q ? v - q : v;
            }
          });
        }
        return out;
      case 'pmul': {
        const block = n / COMPACT_WORDS;
        for (let c = 0; c <= a.deg; c++) {
          each((i, q, iq) => {
            const x = R(a, c, i);
            const o = W(c, i);
            for (let j = 0; j < n; j++) o[j] = mulmodInv(x[j], blob[op.pt + i * COMPACT_WORDS + Math.floor(j / block)], q, iq);
          });
        }
        return out;
      }
      case 'tensor':
        each((i, q, iq) => {
          const a0 = R(a, 0, i);
          const a1 = R(a, 1, i);
          const b0 = R(b, 0, i);
          const b1 = R(b, 1, i);
          const o0 = W(0, i);
          const o1 = W(1, i);
          const o2 = W(2, i);
          for (let j = 0; j < n; j++) {
            o0[j] = mulmodInv(a0[j], b0[j], q, iq);
            const v = mulmodInv(a0[j], b1[j], q, iq) + mulmodInv(a1[j], b0[j], q, iq);
            o1[j] = v >= q ? v - q : v;
            o2[j] = mulmodInv(a1[j], b1[j], q, iq);
          }
        });
        return out;
      case 'relin': {
        if (!this.keys.relin) throw new Error('no relinearization key loaded');
        const ks = this.keySwitch(range(0, lim).map((i) => R(a, 2, i)), l, this.keys.relin);
        for (let c = 0; c < 2; c++) {
          each((i, q, iq) => {
            const x = R(a, c, i);
            const o = W(c, i);
            const k = ks[c][i];
            for (let j = 0; j < n; j++) {
              const v = x[j] + k[j];
              o[j] = v >= q ? v - q : v;
            }
          });
        }
        return out;
      }
      case 'rotate': {
        const key = this.keys.galois.get(op.gal);
        if (!key) throw new Error(`no Galois key for element ${op.gal}`);
        let idx = this.perm.get(op.gal);
        if (!idx) {
          idx = automorphismIndex(this.p.logN, op.gal);
          this.perm.set(op.gal, idx);
        }
        const ks = this.keySwitch(range(0, lim).map((i) => R(a, 1, i)), l, key);
        each((i, q, iq) => {
          const x = R(a, 0, i);
          const k0 = ks[0][i];
          const k1 = ks[1][i];
          const o0 = W(0, i);
          const o1 = W(1, i);
          for (let j = 0; j < n; j++) {
            const t = idx[j];
            const v = x[t] + k0[t];
            o0[j] = v >= q ? v - q : v;
            o1[j] = k1[t];
          }
        });
        return out;
      }
      case 'rescale': {
        const ql = this.moduli[l];
        const h = (ql - 1) >>> 1;
        let invs = this.rescaleInv.get(l);
        if (!invs) {
          invs = range(0, l).map((i) => invmod(ql % this.moduli[i], this.moduli[i]));
          this.rescaleInv.set(l, invs);
        }
        for (let c = 0; c <= a.deg; c++) {
          const last = this.mods[l].intt(Float64Array.from(R(a, c, l)));
          for (let i = 0; i < l; i++) {
            const q = this.moduli[i];
            const row = new Float64Array(n);
            const hq = h % q;
            for (let j = 0; j < n; j++) {
              let t = last[j] + h;
              if (t >= ql) t -= ql;
              const v = (t % q) - hq;
              row[j] = v < 0 ? v + q : v;
            }
            this.mods[i].ntt(row);
            const x = R(a, c, i);
            const o = W(c, i);
            for (let j = 0; j < n; j++) {
              let v = x[j] - row[j];
              if (v < 0) v += q;
              o[j] = mulmodInv(v, invs[i], q, 1 / q);
            }
          }
        }
        return out;
      }
      default:
        throw new Error(`unknown rns op ${op.op}`);
    }
  }
}
