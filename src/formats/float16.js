// IEEE 754 binary16 <-> binary32 scalar conversion. Encoding rounds to
// nearest, ties to even; decoding is exact.
const f32ToF16ScratchF32 = new Float32Array(1);
const f32ToF16ScratchU32 = new Uint32Array(f32ToF16ScratchF32.buffer);

/** @type {import('./float16.js').float32ToFloat16} */
export function float32ToFloat16(value) {
  f32ToF16ScratchF32[0] = value;
  const f = f32ToF16ScratchU32[0];

  const sign = (f >> 31) & 0x1;
  let exp = (f >> 23) & 0xff;
  const frac = f & 0x7fffff;

  if (exp === 0xff) {
    return (sign << 15) | 0x7c00 | (frac ? 0x200 : 0);
  }

  if (exp === 0) {
    return sign << 15;
  }

  exp = exp - 127 + 15;

  if (exp >= 31) {
    return (sign << 15) | 0x7c00;
  }

  if (exp <= 0) {
    if (exp < -10) {
      return sign << 15;
    }
    const significand = frac | 0x800000;
    const shift = 14 - exp;
    let rounded = significand >>> shift;
    const remainder = significand & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (rounded & 1))) {
      rounded += 1;
    }
    return (sign << 15) | rounded;
  }

  // Round to nearest, ties to even. Carry may advance the exponent or overflow
  // the largest finite half to infinity; truncation biases converted weights.
  let rounded = (exp << 10) | (frac >>> 13);
  const remainder = frac & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (rounded & 1))) {
    rounded += 1;
  }
  return (sign << 15) | rounded;
}

/** @type {import('./float16.js').float16ToFloat32} */
export function float16ToFloat32(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) {
      return sign ? -0 : 0;
    }
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }

  if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/** @type {import('./float16.js').decodeFloat16Array} */
export function decodeFloat16Array(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = float16ToFloat32(values[i]);
  }
  return out;
}
