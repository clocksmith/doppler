/** Independent scalar oracle; never imported by converter or runtime owners. */
const MIN_NORMAL = 2 ** -126;
const MAX_K = 133144;

function assert(ok, message) { if (!ok) throw new Error(message); }
function bitsOf(value) {
  const bytes = new ArrayBuffer(4), view = new DataView(bytes);
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}
function fromBits(bits) {
  const bytes = new ArrayBuffer(4), view = new DataView(bytes);
  view.setUint32(0, bits, true);
  return view.getFloat32(0, true);
}
function fraction(value) {
  const bits = bitsOf(Math.abs(value)), exponent = (bits >>> 23) & 255;
  const mantissa = BigInt((bits & 0x7fffff) + (exponent ? 0x800000 : 0));
  const power = exponent ? exponent - 150 : -149;
  return power >= 0 ? [mantissa << BigInt(power), 1n] : [mantissa, 1n << BigInt(-power)];
}
function nearestEven(numerator, denominator) {
  const quotient = numerator / denominator, remainder = numerator % denominator;
  const twice = remainder * 2n;
  return quotient + (twice > denominator || twice === denominator && quotient % 2n ? 1n : 0n);
}
function normalF32(numerator, denominator) {
  let exponent = numerator.toString(2).length - denominator.toString(2).length;
  if (exponent >= 0 ? numerator < denominator << BigInt(exponent)
    : numerator << BigInt(-exponent) < denominator) exponent--;
  assert(exponent >= -126, 'W8A8 scale is subnormal');
  const shift = 23 - exponent;
  let significand = shift >= 0
    ? nearestEven(numerator << BigInt(shift), denominator)
    : nearestEven(numerator, denominator << BigInt(-shift));
  if (significand === 1n << 24n) { exponent++; significand >>= 1n; }
  assert(exponent <= 127, 'W8A8 scale overflow');
  return fromBits(((exponent + 127) << 23) | Number(significand - (1n << 23n)));
}
function dimensions(rows, K) {
  assert(Number.isSafeInteger(rows) && rows > 0 && Number.isSafeInteger(K) && K > 0 && K <= MAX_K,
    'W8A8 dimensions exceed exact I32 accumulation contract');
  const paddedK = Math.ceil(K / 4) * 4, rowStride = 4 + paddedK;
  assert(Number.isSafeInteger(rows * rowStride), 'W8A8 byte size overflow');
  return { paddedK, rowStride };
}

export function quantizeW8A8Reference(values, rows, K) {
  const { paddedK, rowStride } = dimensions(rows, K);
  assert(values instanceof Float32Array && values.length === rows * K, 'W8A8 reference requires exact F32 source shape');
  const bytes = new Uint8Array(rows * rowStride), view = new DataView(bytes.buffer);
  const scales = new Float32Array(rows), codes = new Int8Array(rows * paddedK);
  for (let row = 0; row < rows; row++) {
    let maximum = 0;
    for (let k = 0; k < K; k++) {
      const value = values[row * K + k];
      assert(Number.isFinite(value) && (value === 0 || Math.abs(value) >= MIN_NORMAL), 'W8A8 source is nonfinite or subnormal');
      maximum = Math.max(maximum, Math.abs(value));
    }
    const [maxN, maxD] = fraction(maximum);
    const scale = maximum === 0 ? 1 : normalF32(maxN, maxD * 127n);
    scales[row] = scale; view.setFloat32(row * rowStride, scale, true);
    const [scaleN, scaleD] = fraction(scale);
    for (let k = 0; k < K; k++) {
      const value = values[row * K + k], [n, d] = fraction(value);
      const magnitude = Math.min(127, Number(nearestEven(n * scaleD, d * scaleN)));
      const code = value < 0 ? -magnitude : magnitude;
      codes[row * paddedK + k] = code;
      view.setInt8(row * rowStride + 4 + k, code);
    }
  }
  return { bytes, scales, codes, rowStride, paddedK };
}

function multiplyF32(a, b) {
  const value = Math.fround(a * b);
  assert(Number.isFinite(value), 'W8A8 rescale overflow');
  return value !== 0 && Math.abs(value) < MIN_NORMAL ? (value < 0 ? -0 : 0) : value;
}
export function rescaleW8A8Reference(accumulator, activationScale, weightScale) {
  assert(Number.isInteger(accumulator) && accumulator >= -2147483648 && accumulator <= 2147483647,
    'W8A8 accumulator must be I32');
  for (const scale of [activationScale, weightScale]) {
    assert(Number.isFinite(scale) && scale >= MIN_NORMAL && Math.fround(scale) === scale, 'W8A8 scale must be normal F32');
  }
  const activationScaled = multiplyF32(Math.fround(accumulator), activationScale);
  return { activationScaled, output: multiplyF32(activationScaled, weightScale) };
}
function readRows(bytes, rows, K) {
  const { paddedK, rowStride } = dimensions(rows, K);
  assert(bytes instanceof Uint8Array && bytes.byteLength === rows * rowStride, 'W8A8 stored row byte size differs');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const scales = new Float32Array(rows), codes = new Int8Array(rows * K);
  for (let row = 0; row < rows; row++) {
    const scale = view.getFloat32(row * rowStride, true);
    assert(Number.isFinite(scale) && scale >= MIN_NORMAL, 'W8A8 stored scale is invalid');
    scales[row] = scale;
    for (let k = 0; k < paddedK; k++) {
      const code = view.getInt8(row * rowStride + 4 + k);
      assert(code !== -128 && (k < K || code === 0), 'W8A8 stored code or tail padding is invalid');
      if (k < K) codes[row * K + k] = code;
    }
  }
  return { scales, codes };
}
export function matmulW8A8Reference(activationBytes, weightBytes, M, N, K) {
  const a = readRows(activationBytes, M, K), w = readRows(weightBytes, N, K);
  const accumulators = new Int32Array(M * N), activationScaled = new Float32Array(M * N), output = new Float32Array(M * N);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let sum = 0;
    for (let k = 0; k < K; k++) sum += a.codes[m * K + k] * w.codes[n * K + k];
    const index = m * N + n;
    accumulators[index] = sum;
    const scaled = rescaleW8A8Reference(sum, a.scales[m], w.scales[n]);
    activationScaled[index] = scaled.activationScaled; output[index] = scaled.output;
  }
  return { accumulators, activationScaled, output };
}
