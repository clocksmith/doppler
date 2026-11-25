/**
 * Z_(2^64) fixed-point encoding, sharing and recombination for the secret-shared requester, in plain JS.
 *
 * A ring element is two u32 words (lo, hi), stored little-endian in a Uint32Array so the array's bytes are the wire's
 * u64 LE layout. Encoding needs |round(x * 2^fx)| < 2^53 so the integer is exact in a double; decoding converts the
 * signed 64-bit sum to float64 with one rounding (hi * 2^32 is exact, + lo rounds once), as numpy's int64 -> float64
 * does, then to float32 with Math.fround. So with 64-bit returns every step is bit-identical to the Python requester.
 *
 * Packed returns (operator option `returnShift` S): each operator returns bits S..63 of its share in (64 - S) / 8 bytes.
 * The requester adds the two mod 2^(64-S), sign-extends, and estimates y as (sum + 1) * 2^S: the unknown carry out of
 * the dropped low bits is 0 or 1, so the estimate is within 2^S of y.
 */
const TWO32 = 4294967296, INV32 = 1 / TWO32;

if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) throw Error('the ring code assumes a little-endian platform');

/** round half to even, as numpy's rint */
export function rint(v) {
  const r = Math.round(v);
  return Math.abs(r - v) === 0.5 ? 2 * Math.round(v / 2) : r;
}

/**
 * Encodes x (float32 values) at 2^fx and subtracts the PRG share x0 (u32 words), writing share 1 = enc - x0 into out.
 * Returns the largest |enc| seen, for the range check.
 */
export function encodeAndShare(x, fx, x0, out) {
  const scale = 2 ** fx;
  let maxAbs = 0;
  for (let i = 0, j = 0; i < x.length; i++, j += 2) {
    const v = x[i] * scale;
    let e = Math.round(v);
    if (e - v === 0.5 && (e & 1) !== 0) e -= 1;  // Math.round sends ties up; numpy's rint sends them to even
    const a = e < 0 ? -e : e;
    if (a > maxAbs) maxAbs = a;
    const hi = Math.floor(e * INV32);              // signed high word; exact for |e| < 2^53
    const l = e - hi * TWO32 - x0[j];
    out[j] = l;                                    // Uint32Array stores wrap mod 2^32
    out[j + 1] = hi - x0[j + 1] - (l < 0 ? 1 : 0);
  }
  if (!(maxAbs < 2 ** 53)) throw Error('input outside the exact fixed-point range');
  return maxAbs;
}

/** Plain encoding (no sharing), for tests and the parity check. */
export function encode(x, fx) {
  const out = new Uint32Array(2 * x.length);
  encodeAndShare(x, fx, new Uint32Array(2 * x.length), out);
  return out;
}

/**
 * Adds two packed share vectors (width bytes per element: 8 for full 64-bit shares, 5 for returnShift 24, ...) and
 * decodes to float32 at scale 2^-(fx+fw), into out. Returns the largest |value| for the range check.
 */
export function recombine(a, b, width, shift, fx, fw, out) {
  const n = out.length;
  if (a.length < n * width || b.length < n * width) throw Error('share payload too short');
  const scale = 2 ** (shift - fx - fw);
  const bump = shift ? 1 : 0;
  let maxAbs = 0;
  if (width === 8) {
    const A = aligned32(a, n * 2), B = aligned32(b, n * 2);
    for (let i = 0, j = 0; i < n; i++, j += 2) {
      let lo = A[j] + B[j], carry = 0;
      if (lo >= TWO32) { lo -= TWO32; carry = 1; }
      const hi = (A[j + 1] + B[j + 1] + carry) | 0;
      const v = (hi * TWO32 + lo + bump) * scale;
      const m = v < 0 ? -v : v;
      if (m > maxAbs) maxAbs = m;
      out[i] = v;
    }
    return maxAbs;
  }
  if (width === 5) {
    const da = new DataView(a.buffer, a.byteOffset, a.byteLength), db = new DataView(b.buffer, b.byteOffset, b.byteLength);
    for (let i = 0, p = 0; i < n; i++, p += 5) {
      let lo = da.getUint32(p, true) + db.getUint32(p, true), carry = 0;
      if (lo >= TWO32) { lo -= TWO32; carry = 1; }
      const hi = ((a[p + 4] + b[p + 4] + carry) << 24) >> 24;   // 8-bit high part, sign-extended
      const v = (hi * TWO32 + lo + bump) * scale;
      const m = v < 0 ? -v : v;
      if (m > maxAbs) maxAbs = m;
      out[i] = v;
    }
    return maxAbs;
  }
  if (width <= 6) {
    // Up to 48 bits per element: the whole value fits a double, so add mod 2^(8*width) directly.
    const full = 2 ** (8 * width);
    for (let i = 0, p = 0; i < n; i++, p += width) {
      let va = 0, vb = 0;
      for (let k = width - 1; k >= 0; k--) { va = va * 256 + a[p + k]; vb = vb * 256 + b[p + k]; }
      let s = (va + vb) % full;
      if (s >= full / 2) s -= full;
      const v = (s + bump) * scale;
      const m = v < 0 ? -v : v;
      if (m > maxAbs) maxAbs = m;
      out[i] = v;
    }
    return maxAbs;
  }
  // 7 bytes: a low u32 word and a 24-bit high part with an explicit carry.
  for (let i = 0, p = 0; i < n; i++, p += width) {
    const la = a[p] + a[p + 1] * 256 + a[p + 2] * 65536 + a[p + 3] * 16777216;
    const lb = b[p] + b[p + 1] * 256 + b[p + 2] * 65536 + b[p + 3] * 16777216;
    const ha = a[p + 4] + a[p + 5] * 256 + a[p + 6] * 65536, hb = b[p + 4] + b[p + 5] * 256 + b[p + 6] * 65536;
    const lo = la + lb;
    const carry = lo >= TWO32 ? 1 : 0;
    let hi = (ha + hb + carry) % 16777216;
    if (hi >= 8388608) hi -= 16777216;
    const v = (hi * TWO32 + (lo - carry * TWO32) + bump) * scale;
    const m = v < 0 ? -v : v;
    if (m > maxAbs) maxAbs = m;
    out[i] = v;
  }
  return maxAbs;
}

function aligned32(bytes, words) {
  if (bytes.byteOffset % 4 === 0) return new Uint32Array(bytes.buffer, bytes.byteOffset, words);
  return new Uint32Array(bytes.slice(0, words * 4).buffer);
}
