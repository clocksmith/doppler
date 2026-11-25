// Arbitrary-precision binary floats with the semantics of Go's math/big.Float in its default mode (round to nearest
// even), as Lattigo uses them for CKKS scales, Chebyshev coefficients and deterministic plaintext encoding. Mirroring
// those semantics exactly (including how a result's precision is chosen) lets the JavaScript compiler reproduce
// Lattigo's integer constants and scale decisions bit for bit.
//
// A value is (-1)^neg * m * 2^e with m a non-negative BigInt of at most `prec` bits. Methods mutate the receiver
// like Go's: z.mul(x, y) sets z = x * y and returns z. A receiver of precision 0 takes the larger precision of its
// operands first (Go's rule); a receiver with a precision rounds the exact result to it.

export function bitLength(x) {
  if (x === 0n) return 0;
  const s = x.toString(16);
  return (s.length - 1) * 4 + (32 - Math.clz32(parseInt(s[0], 16)));
}

export class BigFloat {
  constructor(prec = 0) {
    this.prec = prec;
    this.neg = false;
    this.m = 0n;
    this.e = 0;
  }

  static of(x, prec) {
    return new BigFloat(prec ?? 0).setValue(x);
  }

  // Go's bignum.NewFloat(x, prec) and the SetFloat64 / SetInt64 / SetInt / Set family, chosen by the argument type.
  setValue(x) {
    if (x instanceof BigFloat) return this.set(x);
    if (typeof x === 'bigint') return this.setInt(x);
    if (Number.isInteger(x) && !Object.is(x, -0) && Math.abs(x) < 2 ** 63 && this._intHint) return this.setInt64(x);
    return this.setFloat64(x);
  }

  clone() {
    const z = new BigFloat(this.prec);
    z.neg = this.neg;
    z.m = this.m;
    z.e = this.e;
    return z;
  }

  isZero() {
    return this.m === 0n;
  }

  // Rounds (neg, m, e) to this.prec bits; sticky marks a non-zero remainder below m's last bit.
  _store(neg, m, e, sticky = false) {
    const prec = this.prec;
    if (m === 0n) {
      this.neg = neg;
      this.m = 0n;
      this.e = 0;
      return this;
    }
    const bl = bitLength(m);
    if (bl > prec) {
      const shift = bl - prec;
      let r = m >> BigInt(shift);
      const rem = m & ((1n << BigInt(shift)) - 1n);
      const half = 1n << BigInt(shift - 1);
      if (rem > half || (rem === half && (sticky || (r & 1n) === 1n))) r += 1n;
      e += shift;
      if (bitLength(r) > prec) {
        r >>= 1n;
        e += 1;
      }
      m = r;
    } else if (sticky) {
      throw new Error('BigFloat: sticky rounding needs extra bits');
    }
    // Normalize: strip trailing zero bits so equal values have equal representations.
    while (m !== 0n && (m & 1n) === 0n) {
      m >>= 1n;
      e += 1;
    }
    this.neg = neg;
    this.m = m;
    this.e = e;
    return this;
  }

  setPrec(prec) {
    this.prec = prec;
    if (prec === 0) {
      this.m = 0n;
      this.e = 0;
      return this;
    }
    return this._store(this.neg, this.m, this.e);
  }

  set(x) {
    if (this === x) return this;
    if (this.prec === 0) this.prec = x.prec;
    return this._store(x.neg, x.m, x.e);
  }

  // Go's Copy: value and precision.
  copy(x) {
    this.prec = x.prec;
    this.neg = x.neg;
    this.m = x.m;
    this.e = x.e;
    return this;
  }

  setFloat64(f) {
    if (!Number.isFinite(f)) throw new Error('BigFloat: non-finite float64');
    if (this.prec === 0) this.prec = 53;
    if (f === 0) return this._store(Object.is(f, -0), 0n, 0);
    const neg = f < 0;
    const { m, e } = decomposeFloat64(Math.abs(f));
    return this._store(neg, m, e);
  }

  setInt64(i) {
    if (this.prec === 0) this.prec = 64;
    const b = BigInt(i);
    return this._store(b < 0n, b < 0n ? -b : b, 0);
  }

  setInt(b) {
    if (this.prec === 0) this.prec = Math.max(bitLength(b < 0n ? -b : b), 64);
    return this._store(b < 0n, b < 0n ? -b : b, 0);
  }

  setRational(num, den) {
    // num / den correctly rounded to this.prec (which must be set).
    if (this.prec === 0) throw new Error('BigFloat: setRational needs a precision');
    const neg = (num < 0n) !== (den < 0n);
    num = num < 0n ? -num : num;
    den = den < 0n ? -den : den;
    if (num === 0n) return this._store(false, 0n, 0);
    const k = Math.max(0, this.prec + 2 + bitLength(den) - bitLength(num));
    const q = (num << BigInt(k)) / den;
    const r = (num << BigInt(k)) - q * den;
    return this._store(neg, q, -k, r !== 0n);
  }

  _prec2(x, y) {
    if (this.prec === 0) this.prec = Math.max(x.prec, y.prec);
  }

  add(x, y) {
    this._prec2(x, y);
    return this._addSigned(x.neg, x.m, x.e, y.neg, y.m, y.e);
  }

  sub(x, y) {
    this._prec2(x, y);
    return this._addSigned(x.neg, x.m, x.e, !y.neg, y.m, y.e);
  }

  _addSigned(nx, mx, ex, ny, my, ey) {
    if (mx === 0n && my === 0n) return this._store(nx && ny, 0n, 0);
    if (mx === 0n) return this._store(ny, my, ey);
    if (my === 0n) return this._store(nx, mx, ex);
    const e = Math.min(ex, ey);
    const ax = mx << BigInt(ex - e);
    const ay = my << BigInt(ey - e);
    const sx = nx ? -ax : ax;
    const sy = ny ? -ay : ay;
    const s = sx + sy;
    if (s === 0n) return this._store(false, 0n, 0);
    return this._store(s < 0n, s < 0n ? -s : s, e);
  }

  mul(x, y) {
    this._prec2(x, y);
    return this._store(x.neg !== y.neg, x.m * y.m, x.e + y.e);
  }

  quo(x, y) {
    this._prec2(x, y);
    if (y.m === 0n) throw new Error('BigFloat: division by zero');
    if (x.m === 0n) return this._store(x.neg !== y.neg, 0n, 0);
    const k = Math.max(0, this.prec + 2 + bitLength(y.m) - bitLength(x.m));
    const num = x.m << BigInt(k);
    const q = num / y.m;
    const r = num - q * y.m;
    return this._store(x.neg !== y.neg, q, x.e - y.e - k, r !== 0n);
  }

  neg_(x) {
    this.set(x);
    this.neg = !this.neg;
    return this;
  }

  abs_(x) {
    this.set(x);
    this.neg = false;
    return this;
  }

  cmp(y) {
    return cmpBF(this, y);
  }

  sign() {
    return this.m === 0n ? 0 : this.neg ? -1 : 1;
  }

  // Truncation toward zero (Go's Float.Int).
  toBigInt() {
    if (this.m === 0n) return 0n;
    const mag = this.e >= 0 ? this.m << BigInt(this.e) : this.m >> BigInt(-this.e);
    return this.neg ? -mag : mag;
  }

  isInt() {
    if (this.m === 0n || this.e >= 0) return true;
    return (this.m & ((1n << BigInt(-this.e)) - 1n)) === 0n;
  }

  toNumber() {
    if (this.m === 0n) return this.neg ? -0 : 0;
    const t = new BigFloat(53)._store(this.neg, this.m, this.e);
    const v = Number(t.m) * 2 ** t.e;
    if (!Number.isFinite(v) || (v === 0 && t.m !== 0n)) {
      // Split the power of two to stay within range.
      const h = Math.trunc(t.e / 2);
      return (t.neg ? -1 : 1) * Number(t.m) * 2 ** h * 2 ** (t.e - h);
    }
    return t.neg ? -v : v;
  }

  // Go's x.Text('p', 0) with the precision, as the golden fixtures record values: "<prec>|<text>".
  toGoText() {
    if (this.m === 0n) return `${this.prec}|${this.neg ? '-' : ''}0`;
    const bl = bitLength(this.m);
    // mantissa in [0.5, 1): m * 2^e = (m / 2^bl) * 2^(e + bl); hex digits of m left-aligned to a multiple of 4 bits.
    const pad = (4 - (bl % 4)) % 4;
    let hex = (this.m << BigInt(pad)).toString(16).replace(/0+$/, '');
    const exp = this.e + bl;
    return `${this.prec}|${this.neg ? '-' : ''}0x.${hex}p${exp >= 0 ? '+' : ''}${exp}`;
  }

  static fromGoText(s) {
    const [p, t] = s.split('|');
    const z = new BigFloat(Number(p));
    const neg = t.startsWith('-');
    const body = neg ? t.slice(1) : t;
    if (body === '0') return z._store(neg, 0n, 0);
    const mt = /^0x\.([0-9a-f]+)p([+-]\d+)$/.exec(body);
    if (!mt) throw new Error(`BigFloat: cannot parse ${s}`);
    const m = BigInt(`0x${mt[1]}`);
    z.neg = neg;
    z.m = m;
    z.e = Number(mt[2]) - 4 * mt[1].length;
    while (z.m !== 0n && (z.m & 1n) === 0n) {
      z.m >>= 1n;
      z.e += 1;
    }
    return z;
  }
}

export function cmpBF(x, y) {
  const sx = x.m === 0n ? 0 : x.neg ? -1 : 1;
  const sy = y.m === 0n ? 0 : y.neg ? -1 : 1;
  if (sx !== sy) return sx < sy ? -1 : 1;
  if (sx === 0) return 0;
  const e = Math.min(x.e, y.e);
  const ax = x.m << BigInt(x.e - e);
  const ay = y.m << BigInt(y.e - e);
  const c = ax === ay ? 0 : ax < ay ? -1 : 1;
  return sx < 0 ? -c : c;
}

// Exact (m, e) with f = m * 2^e for a positive finite float64.
export function decomposeFloat64(f) {
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, f);
  const hi = buf.getUint32(0);
  const lo = buf.getUint32(4);
  const exp = (hi >>> 20) & 0x7ff;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e;
  if (exp === 0) {
    e = -1074;
  } else {
    m |= 1n << 52n;
    e = exp - 1075;
  }
  while (m !== 0n && (m & 1n) === 0n) {
    m >>= 1n;
    e += 1;
  }
  return { m, e };
}

// Go's bignum.NewFloat(x, prec): precision set first, then the value.
export function newFloat(x, prec) {
  const z = new BigFloat(prec);
  if (x instanceof BigFloat) return z.set(x);
  if (typeof x === 'bigint') return z.setInt(x);
  return z.setFloat64(x);
}

export function newInt64(i, prec = 0) {
  return new BigFloat(prec).setInt64(i);
}

const PI_DIGITS = '31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989';

// bignum.Pi(prec): the decimal constant correctly rounded to prec bits.
export function pi(prec) {
  return new BigFloat(prec).setRational(BigInt(PI_DIGITS), 10n ** BigInt(PI_DIGITS.length - 1));
}

// bignum.Cos: the halving and doubling formula Lattigo uses, operation for operation.
export function cos(x) {
  const tmp = new BigFloat();
  const t = newFloat(0.5, x.prec);
  const half = new BigFloat().copy(t);
  for (let i = 1; i < (x.prec >> 1) - 1; i++) t.mul(t, half);
  const s = new BigFloat().mul(x, t);
  s.mul(s, x);
  s.mul(s, t);
  const four = newFloat(4.0, x.prec);
  for (let i = 1; i < x.prec >> 1; i++) {
    tmp.sub(four, s);
    s.mul(s, tmp);
  }
  const cosx = new BigFloat().quo(s, newFloat(2.0, x.prec));
  cosx.sub(newFloat(1.0, x.prec), cosx);
  return cosx;
}

// bignum.Complex: a pair of BigFloats. mulComplex mirrors bignum.ComplexMultiplier.Mul (real fast paths included).
export class Complex {
  constructor(re = new BigFloat(), im = new BigFloat()) {
    this.re = re;
    this.im = im;
  }

  static withPrec(prec) {
    return new Complex(new BigFloat(prec), new BigFloat(prec));
  }

  isReal() {
    return this.im === null || this.im.m === 0n;
  }

  clone() {
    return new Complex(new BigFloat().set(this.re), new BigFloat().set(this.im));
  }

  set(a) {
    this.re.set(a.re);
    this.im.set(a.im);
    return this;
  }

  add(a, b) {
    this.re.add(a.re, b.re);
    this.im.add(a.im, b.im);
    return this;
  }

  sub(a, b) {
    this.re.sub(a.re, b.re);
    this.im.sub(a.im, b.im);
    return this;
  }
}

export class ComplexMultiplier {
  constructor() {
    this.t0 = new BigFloat();
    this.t1 = new BigFloat();
    this.t2 = new BigFloat();
    this.t3 = new BigFloat();
  }

  // c = a * b; c may alias a or b.
  mul(a, b, c) {
    if (a.isReal()) {
      if (b.isReal()) {
        c.re.mul(a.re, b.re);
        c.im.setFloat64(0);
      } else {
        c.im.mul(a.re, b.im);
        c.re.mul(a.re, b.re);
      }
    } else if (b.isReal()) {
      c.im.mul(a.im, b.re);
      c.re.mul(a.re, b.re);
    } else {
      this.t0.mul(a.re, b.re);
      this.t1.mul(a.im, b.im);
      this.t2.mul(a.re, b.im);
      this.t3.mul(a.im, b.re);
      c.re.sub(this.t0, this.t1);
      c.im.add(this.t2, this.t3);
    }
  }
}
