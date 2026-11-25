// SplitMix64 in 32-bit halves (no BigInt in the hot path): the input generator shared with the one-time Go export of
// the golden RNS vectors (fixtures/go-golden.json, "rns").

const M16 = 0xffff;

// (a * b) mod 2^64 for a = aH:aL, b = bH:bL (u32 halves).
function mul64(aH, aL, bH, bL) {
  const a0 = aL & M16;
  const a1 = aL >>> 16;
  const b0 = bL & M16;
  const b1 = bL >>> 16;
  const p00 = a0 * b0;
  const p01 = a0 * b1;
  const p10 = a1 * b0;
  const p11 = a1 * b1;
  const mid = (p00 >>> 16) + (p01 & M16) + (p10 & M16);
  const lo = ((mid & M16) << 16) | (p00 & M16);
  let hi = p11 + (p01 >>> 16) + (p10 >>> 16) + (mid >>> 16);
  hi = (hi + Math.imul(aH, bL) + Math.imul(aL, bH)) >>> 0;
  return [hi, lo >>> 0];
}

export class SplitMix64 {
  constructor(seed) {
    const s = BigInt.asUintN(64, BigInt(seed));
    this.h = Number(s >> 32n);
    this.l = Number(s & 0xffffffffn);
  }

  // Next 64-bit output as [hi, lo].
  next() {
    // s += 0x9e3779b97f4a7c15
    let lo = this.l + 0x7f4a7c15;
    const carry = lo >= 2 ** 32 ? 1 : 0;
    lo >>>= 0;
    this.l = lo;
    this.h = (this.h + 0x9e3779b9 + carry) >>> 0;
    let zh = this.h;
    let zl = this.l;
    // z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
    let sl = ((zl >>> 30) | (zh << 2)) >>> 0;
    let sh = zh >>> 30;
    [zh, zl] = mul64((zh ^ sh) >>> 0, (zl ^ sl) >>> 0, 0xbf58476d, 0x1ce4e5b9);
    // z = (z ^ (z >> 27)) * 0x94d049bb133111eb
    sl = ((zl >>> 27) | (zh << 5)) >>> 0;
    sh = zh >>> 27;
    [zh, zl] = mul64((zh ^ sh) >>> 0, (zl ^ sl) >>> 0, 0x94d049bb, 0x133111eb);
    // z ^ (z >> 31)
    sl = ((zl >>> 31) | (zh << 1)) >>> 0;
    sh = zh >>> 31;
    return [(zh ^ sh) >>> 0, (zl ^ sl) >>> 0];
  }

  // next() mod q for q < 2^32.
  below(q) {
    const [h, l] = this.next();
    // (h * 2^32 + l) mod q
    const t = ((h % q) * 65536) % q;
    return ((t * 65536) % q + l) % q;
  }
}
