/**
 * CPU-side Q4_K_M dequantization for vision encoder weights.
 *
 * Q4_K_M layout per 256-element super-block (144 bytes):
 *   - 2 bytes: f16 scale (d)
 *   - 2 bytes: f16 min (dmin)
 *   - 12 bytes: 6-bit scale/min pairs for 8 sub-blocks (K_SCALE_SIZE)
 *   - 128 bytes: 4-bit quantized values (256 nibbles packed into 128 bytes)
 *
 * Total: 2 + 2 + 12 + 128 = 144 bytes per block of 256 elements.
 */

const QK_K = 256;
const Q4K_BLOCK_BYTES = 144;

function f16ToF32(u16) {
  const sign = (u16 >>> 15) & 1;
  const exp = (u16 >>> 10) & 0x1f;
  const frac = u16 & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 31) {
    return frac === 0
      ? (sign ? -Infinity : Infinity)
      : NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function getScaleMin(scales, j, isMin) {
  if (j < 4) {
    const idx = isMin ? j + 4 : j;
    return scales[idx] & 0x3f;
  }
  const base = 8 + j - 4;
  const highBits = (scales[base] >>> (isMin ? 4 : 0)) & 0x0f;
  const lowIdx = isMin ? j + 4 - 4 : j;
  const lowBits = scales[lowIdx] >>> 6;
  return lowBits | (highBits << 2);
}

export function dequantQ4KBlock(blockBytes, offset) {
  const view = new DataView(blockBytes.buffer, blockBytes.byteOffset + offset, Q4K_BLOCK_BYTES);
  const d = f16ToF32(view.getUint16(0, true));
  const dmin = f16ToF32(view.getUint16(2, true));
  const scales = new Uint8Array(blockBytes.buffer, blockBytes.byteOffset + offset + 4, 12);
  const qs = new Uint8Array(blockBytes.buffer, blockBytes.byteOffset + offset + 16, 128);

  const out = new Float32Array(QK_K);

  for (let j = 0; j < QK_K / 64; j++) {
    const sc = getScaleMin(scales, j, false);
    const m = getScaleMin(scales, j, true);
    const dsc = d * sc;
    const dm = dmin * m;

    for (let l = 0; l < 32; l++) {
      const qsByte = qs[32 * j + l];
      out[64 * j + l] = dsc * (qsByte & 0x0f) - dm;
      out[64 * j + l + 32] = dsc * (qsByte >>> 4) - dm;
    }
  }

  return out;
}

export function dequantQ4K(data, numElements) {
  const numBlocks = Math.ceil(numElements / QK_K);
  const result = new Float32Array(numElements);
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);

  for (let b = 0; b < numBlocks; b++) {
    const blockOut = dequantQ4KBlock(bytes, b * Q4K_BLOCK_BYTES);
    const remaining = Math.min(QK_K, numElements - b * QK_K);
    result.set(blockOut.subarray(0, remaining), b * QK_K);
  }

  return result;
}

export function bf16ToF32Array(data, numElements) {
  const u16 = new Uint16Array(data.buffer, data.byteOffset, numElements);
  const out = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    const bits = u16[i] << 16;
    const view = new DataView(new ArrayBuffer(4));
    view.setInt32(0, bits);
    out[i] = view.getFloat32(0);
  }
  return out;
}

export function toF32(tensorData, dtype, numElements) {
  if (tensorData instanceof Float32Array) {
    return tensorData;
  }
  const raw = tensorData instanceof Uint8Array
    ? tensorData
    : new Uint8Array(tensorData.buffer, tensorData.byteOffset, tensorData.byteLength);

  const normalizedDtype = (typeof dtype === 'string' ? dtype : '').toUpperCase();

  if (normalizedDtype === 'Q4_K_M' || normalizedDtype === 'Q4_K') {
    return dequantQ4K(raw, numElements);
  }
  if (normalizedDtype === 'BF16') {
    return bf16ToF32Array(raw, numElements);
  }
  if (normalizedDtype === 'F16') {
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, numElements);
    const out = new Float32Array(numElements);
    for (let i = 0; i < numElements; i++) {
      out[i] = f16ToF32(u16[i]);
    }
    return out;
  }
  if (normalizedDtype === 'F32') {
    return new Float32Array(raw.buffer, raw.byteOffset, numElements);
  }

  throw new Error(`[Vision] Unsupported dtype for CPU dequant: "${dtype}"`);
}
