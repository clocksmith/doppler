/**
 * Reference Dequantization Implementation
 */

// ============================================================================
// Helpers (fp16)
// ============================================================================

/**
 * Convert fp16 bits to fp32
 */
function float16ToFloat32(bits: number): number {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1F;
  const frac = bits & 0x3FF;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Denormalized
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }

  if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

export function float32ToFloat16(value: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = value;
  const f = int32View[0];

  const sign = (f >> 31) & 0x1;
  let exp = (f >> 23) & 0xff;
  let frac = f & 0x7fffff;

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
    frac = (frac | 0x800000) >> (1 - exp);
    return (sign << 15) | (frac >> 13);
  }

  return (sign << 15) | (exp << 10) | (frac >> 13);
}

/**
 * Reference INT8 dequantization
 * float = (int8 - zero_point) * scale
 *
 * @param quantized Quantized values
 * @param scales Per-channel or per-tensor scales
 * @param zeroPoints Zero points (optional, default 0)
 * @param numChannels Number of output channels
 * @param channelSize Elements per channel
 * @returns Dequantized values
 */
export function dequantInt8Ref(
  quantized: Int8Array,
  scales: Float32Array,
  zeroPoints: Int8Array | null = null,
  numChannels: number = 1,
  channelSize: number = 0
): Float32Array {
  const output = new Float32Array(quantized.length);

  if (channelSize === 0) {
    channelSize = quantized.length / numChannels;
  }

  for (let c = 0; c < numChannels; c++) {
    const scale = scales[c];
    const zp = zeroPoints ? zeroPoints[c] : 0;

    for (let i = 0; i < channelSize; i++) {
      const idx = c * channelSize + i;
      output[idx] = (quantized[idx] - zp) * scale;
    }
  }

  return output;
}

/**
 * Reference INT4 dequantization (packed 2 values per byte)
 * @param quantized Packed INT4 values (2 per byte)
 * @param scales Scales
 * @param numElements Total number of output elements
 * @param groupSize Group size for quantization
 * @returns Dequantized values
 */
export function dequantInt4Ref(
  quantized: Uint8Array,
  scales: Float32Array,
  numElements: number,
  groupSize: number = 32
): Float32Array {
  const output = new Float32Array(numElements);
  const numGroups = Math.ceil(numElements / groupSize);

  for (let i = 0; i < numElements; i++) {
    const byteIdx = Math.floor(i / 2);
    const groupIdx = Math.floor(i / groupSize);
    const scale = scales[groupIdx];

    let val: number;
    if (i % 2 === 0) {
      // Low nibble
      val = quantized[byteIdx] & 0x0F;
    } else {
      // High nibble
      val = (quantized[byteIdx] >> 4) & 0x0F;
    }

    // Convert from unsigned [0,15] to signed [-8,7]
    if (val >= 8) {
      val = val - 16;
    }

    output[i] = val * scale;
  }

  return output;
}

/**
 * Reference block-wise quantization (Q4_0 style)
 * Each block has: scale (fp16) + 32 int4 values (16 bytes)
 */
export function dequantQ4_0Ref(quantized: Uint8Array, numBlocks: number): Float32Array {
  const blockSize = 32;
  const output = new Float32Array(numBlocks * blockSize);
  const dataView = new DataView(quantized.buffer);

  for (let block = 0; block < numBlocks; block++) {
    // Q4_0 block: 2 bytes scale (fp16) + 16 bytes data (32 int4)
    const blockOffset = block * 18;

    // Read scale as fp16 (simplified - just use the bytes directly for now)
    const scaleBytes = dataView.getUint16(blockOffset, true);
    const scale = float16ToFloat32(scaleBytes);

    // Unpack 32 int4 values from 16 bytes
    for (let i = 0; i < 16; i++) {
      const byte = quantized[blockOffset + 2 + i];

      const low = (byte & 0x0F) - 8;
      const high = ((byte >> 4) & 0x0F) - 8;

      output[block * blockSize + i * 2] = low * scale;
      output[block * blockSize + i * 2 + 1] = high * scale;
    }
  }

  return output;
}

// ============================================================================
// Q4_K (llama.cpp block_q4_K) reference
// ============================================================================

const Q4K_K = 256;
const Q4K_BLOCK_SIZE = 144;

function findMinMax(data: Float32Array, offset: number, length: number): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    const val = data[offset + i];
    if (val < min) min = val;
    if (val > max) max = val;
  }
  return { min, max };
}

/**
 * Quantize a single Q4_K block (256 values) into llama.cpp Q4_K byte layout.
 * This matches the dequant_* kernels in `doppler/gpu/kernels/dequant_*.wgsl`.
 */
export function quantizeQ4_KBlockRef(data: Float32Array, offset: number): Uint8Array {
  const block = new Uint8Array(Q4K_BLOCK_SIZE);
  const view = new DataView(block.buffer);

  const scales = new Float32Array(8);
  const minOffsets = new Float32Array(8);
  const qs = new Uint8Array(256);

  for (let sb = 0; sb < 8; sb++) {
    const sbOffset = offset + sb * 32;
    const { min, max } = findMinMax(data, sbOffset, 32);

    minOffsets[sb] = -min;
    const range = max - min;
    scales[sb] = range > 0 ? range / 15 : 0;

    const invScale = scales[sb] > 0 ? 1 / scales[sb] : 0;
    for (let i = 0; i < 32; i++) {
      const val = data[sbOffset + i];
      let q = Math.round((val - min) * invScale);
      q = Math.max(0, Math.min(15, q));
      qs[sb * 32 + i] = q;
    }
  }

  let maxScale = 0;
  let maxMinOffset = 0;
  for (let i = 0; i < 8; i++) {
    if (scales[i] > maxScale) maxScale = scales[i];
    if (minOffsets[i] > maxMinOffset) maxMinOffset = minOffsets[i];
    if (minOffsets[i] < 0) minOffsets[i] = 0;
  }

  const d = maxScale / 63;
  const dmin = maxMinOffset / 63;

  view.setUint16(0, float32ToFloat16(d), true);
  view.setUint16(2, float32ToFloat16(dmin), true);

  const invD = d > 0 ? 1 / d : 0;
  const invDmin = dmin > 0 ? 1 / dmin : 0;

  const scaleBits = new Uint8Array(8);
  const minBits = new Uint8Array(8);
  for (let i = 0; i < 8; i++) {
    scaleBits[i] = Math.min(63, Math.round(scales[i] * invD));
    minBits[i] = Math.min(63, Math.round(Math.max(0, minOffsets[i]) * invDmin));
  }

  // bytes 0-3: low 6 bits of scales[0..3], high 2 bits from scales[4..7]
  for (let i = 0; i < 4; i++) {
    const scaleLo = scaleBits[i] & 0x3f;
    const scaleHi2 = (scaleBits[i + 4] >> 4) & 0x03;
    block[4 + i] = scaleLo | (scaleHi2 << 6);
  }

  // bytes 4-7: low 6 bits of mins[0..3], high 2 bits from mins[4..7]
  for (let i = 0; i < 4; i++) {
    const minLo = minBits[i] & 0x3f;
    const minHi2 = (minBits[i + 4] >> 4) & 0x03;
    block[4 + 4 + i] = minLo | (minHi2 << 6);
  }

  // bytes 8-11: low 4 bits scales[4..7] and mins[4..7]
  for (let i = 0; i < 4; i++) {
    const scaleLo4 = scaleBits[i + 4] & 0x0f;
    const minLo4 = minBits[i + 4] & 0x0f;
    block[4 + 8 + i] = scaleLo4 | (minLo4 << 4);
  }

  // qs: 4 chunks of 64 elements, packed into 32 bytes each (lo nibbles then hi nibbles)
  for (let chunk = 0; chunk < 4; chunk++) {
    const chunkBase = chunk * 64;
    const byteBase = 16 + chunk * 32;
    for (let i = 0; i < 32; i++) {
      const lo = qs[chunkBase + i] & 0x0f;
      const hi = qs[chunkBase + 32 + i] & 0x0f;
      block[byteBase + i] = lo | (hi << 4);
    }
  }

  return block;
}

export function quantizeQ4_KRef(values: Float32Array, numBlocks: number): Uint8Array {
  const out = new Uint8Array(numBlocks * Q4K_BLOCK_SIZE);
  for (let b = 0; b < numBlocks; b++) {
    const block = quantizeQ4_KBlockRef(values, b * Q4K_K);
    out.set(block, b * Q4K_BLOCK_SIZE);
  }
  return out;
}

export function dequantizeQ4_KBlockRef(block: Uint8Array): Float32Array {
  const view = new DataView(block.buffer, block.byteOffset);
  const out = new Float32Array(Q4K_K);

  const d = float16ToFloat32(view.getUint16(0, true));
  const dmin = float16ToFloat32(view.getUint16(2, true));

  const scaleBits = new Uint8Array(8);
  const minBits = new Uint8Array(8);

  for (let i = 0; i < 4; i++) {
    scaleBits[i] = block[4 + i] & 0x3f;
    scaleBits[i + 4] = ((block[4 + i] >> 6) & 0x03) << 4;
  }
  for (let i = 0; i < 4; i++) {
    minBits[i] = block[4 + 4 + i] & 0x3f;
    minBits[i + 4] = ((block[4 + 4 + i] >> 6) & 0x03) << 4;
  }
  for (let i = 0; i < 4; i++) {
    scaleBits[i + 4] |= block[4 + 8 + i] & 0x0f;
    minBits[i + 4] |= (block[4 + 8 + i] >> 4) & 0x0f;
  }

  const scales = new Float32Array(8);
  const mins = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    scales[i] = d * scaleBits[i];
    mins[i] = dmin * minBits[i];
  }

  for (let chunk = 0; chunk < 4; chunk++) {
    const chunkBase = chunk * 64;
    const byteBase = 16 + chunk * 32;
    for (let i = 0; i < 32; i++) {
      const byte = block[byteBase + i];
      const lo = byte & 0x0f;
      const hi = (byte >> 4) & 0x0f;

      const sb0 = Math.floor((chunkBase + i) / 32);
      const sb1 = Math.floor((chunkBase + 32 + i) / 32);

      out[chunkBase + i] = scales[sb0] * lo - mins[sb0];
      out[chunkBase + 32 + i] = scales[sb1] * hi - mins[sb1];
    }
  }

  return out;
}

export function dequantQ4_KRef(quantized: Uint8Array, numBlocks: number): Float32Array {
  const out = new Float32Array(numBlocks * Q4K_K);
  for (let b = 0; b < numBlocks; b++) {
    const start = b * Q4K_BLOCK_SIZE;
    const block = quantized.subarray(start, start + Q4K_BLOCK_SIZE);
    out.set(dequantizeQ4_KBlockRef(block), b * Q4K_K);
  }
  return out;
}

export default dequantInt8Ref;
