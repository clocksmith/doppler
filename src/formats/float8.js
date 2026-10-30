/** @type {typeof import('./float8.js').float8ToFloat32} */
export function float8ToFloat32(code, encoding) {
  if (!Number.isInteger(code) || code < 0 || code > 255) throw new Error('FP8 code must be an unsigned byte.');
  if (encoding !== 'e4m3fn' && encoding !== 'e5m2') throw new Error(`Unsupported FP8 encoding ${encoding}.`);
  const mantissaBits = encoding === 'e4m3fn' ? 3 : 2;
  const bias = encoding === 'e4m3fn' ? 7 : 15;
  const magnitude = code & 127;
  const exponent = magnitude >>> mantissaBits;
  const mantissa = magnitude & ((1 << mantissaBits) - 1);
  if (encoding === 'e4m3fn' && magnitude === 127) return NaN;
  if (encoding === 'e5m2' && exponent === 31) {
    if (mantissa) return NaN;
    return code & 128 ? -Infinity : Infinity;
  }
  const value = exponent === 0
    ? mantissa * 2 ** (1 - bias - mantissaBits)
    : (1 + mantissa / (1 << mantissaBits)) * 2 ** (exponent - bias);
  return code & 128 ? -value : value;
}
