

// ============================================================================
// Type Guards
// ============================================================================


export function isContiguousLayer(layer) {
  return 'keys' in layer && 'values' in layer;
}


export function isPagedLayer(layer) {
  return 'keyPages' in layer && 'valuePages' in layer;
}

// ============================================================================
// F16 Conversion Utilities
// ============================================================================

const f32View = new Float32Array(1);
const u32View = new Uint32Array(f32View.buffer);


export function f32ToF16Bits(value) {
  f32View[0] = value;
  const x = u32View[0];
  const sign = (x >> 16) & 0x8000;
  let exp = ((x >> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;

  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >> (1 - exp);
    return sign | ((mant + 0x1000) >> 13);
  }

  if (exp >= 31) {
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }

  return sign | (exp << 10) | ((mant + 0x1000) >> 13);
}


export function f16ToF32Bits(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    const f = mant / 1024 * Math.pow(2, -14);
    return sign ? -f : f;
  }
  if (exp === 31) {
    return mant ? NaN : (sign ? -Infinity : Infinity);
  }

  const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}


export function f32ToF16Array(input) {
  const out = new Uint16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = f32ToF16Bits(input[i]);
  }
  return out;
}


export function f16ToF32Array(input) {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = f16ToF32Bits(input[i]);
  }
  return out;
}
