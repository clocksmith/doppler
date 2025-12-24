/**
 * Tensor Quantization for .rdrr Format
 * Implements Q4_K_M quantization (4-bit with k-means clustering).
 */

const QK_K = 256;
const K_SCALE_SIZE = 12;
const QK4_K_BLOCK_SIZE = 144;

export interface QuantizeResult {
  quantized: Uint8Array;
  numBlocks: number;
  originalSize: number;
  quantizedSize: number;
  compressionRatio: number;
}

export interface QuantizationError {
  mse: number;
  maxError: number;
  snr: number;
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

export function float16ToFloat32(h: number): number {
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

function quantizeQ4KBlock(data: Float32Array, offset: number): Uint8Array {
  const block = new Uint8Array(QK4_K_BLOCK_SIZE);
  const blockView = new DataView(block.buffer);

  const scales = new Float32Array(8);
  const minOffsets = new Float32Array(8);
  const quantized = new Uint8Array(256);

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
      quantized[sb * 32 + i] = q;
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

  blockView.setUint16(0, float32ToFloat16(d), true);
  blockView.setUint16(2, float32ToFloat16(dmin), true);

  const invD = d > 0 ? 1 / d : 0;
  const invDmin = dmin > 0 ? 1 / dmin : 0;

  const scaleBits = new Uint8Array(8);
  const minBits = new Uint8Array(8);

  for (let i = 0; i < 8; i++) {
    scaleBits[i] = Math.min(63, Math.round(scales[i] * invD));
    minBits[i] = Math.min(63, Math.round(Math.max(0, minOffsets[i]) * invDmin));
  }

  for (let i = 0; i < 4; i++) {
    const scale_lo = scaleBits[i] & 0x3f;
    const scale_hi_bits = (scaleBits[i + 4] >> 4) & 0x03;
    block[4 + i] = scale_lo | (scale_hi_bits << 6);
  }

  for (let i = 0; i < 4; i++) {
    const min_lo = minBits[i] & 0x3f;
    const min_hi_bits = (minBits[i + 4] >> 4) & 0x03;
    block[4 + 4 + i] = min_lo | (min_hi_bits << 6);
  }

  for (let i = 0; i < 4; i++) {
    const scale_lo4 = scaleBits[i + 4] & 0x0f;
    const min_lo4 = minBits[i + 4] & 0x0f;
    block[4 + 8 + i] = scale_lo4 | (min_lo4 << 4);
  }

  for (let chunk = 0; chunk < 4; chunk++) {
    const chunkBase = chunk * 64;
    const byteBase = 16 + chunk * 32;
    for (let i = 0; i < 32; i++) {
      const lo = quantized[chunkBase + i] & 0x0f;
      const hi = quantized[chunkBase + 32 + i] & 0x0f;
      block[byteBase + i] = lo | (hi << 4);
    }
  }

  return block;
}

function dequantizeQ4KBlock(block: Uint8Array): Float32Array {
  const blockView = new DataView(block.buffer, block.byteOffset);
  const result = new Float32Array(256);

  const d = float16ToFloat32(blockView.getUint16(0, true));
  const dmin = float16ToFloat32(blockView.getUint16(2, true));

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
  const minOffsets = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    scales[i] = d * scaleBits[i];
    minOffsets[i] = dmin * minBits[i];
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

      result[chunkBase + i] = scales[sb0] * lo - minOffsets[sb0];
      result[chunkBase + 32 + i] = scales[sb1] * hi - minOffsets[sb1];
    }
  }

  return result;
}

/**
 * Flat Q4K quantization (original behavior).
 * Packs all elements sequentially - blocks may cross row boundaries.
 * NOT compatible with fused matmul kernel for non-256-aligned K dimensions.
 */
export function quantizeToQ4KM(data: Float32Array, shape: number[]): QuantizeResult {
  const numElements = shape.reduce((a, b) => a * b, 1);

  if (data.length !== numElements) {
    throw new Error(`Data length ${data.length} doesn't match shape ${shape}`);
  }

  const numBlocks = Math.ceil(numElements / QK_K);
  const paddedLength = numBlocks * QK_K;
  const paddedData = new Float32Array(paddedLength);
  paddedData.set(data);

  const quantized = new Uint8Array(numBlocks * QK4_K_BLOCK_SIZE);

  for (let b = 0; b < numBlocks; b++) {
    const block = quantizeQ4KBlock(paddedData, b * QK_K);
    quantized.set(block, b * QK4_K_BLOCK_SIZE);
  }

  return {
    quantized,
    numBlocks,
    originalSize: numElements * 4,
    quantizedSize: quantized.length,
    compressionRatio: (numElements * 4) / quantized.length,
  };
}

/**
 * Row-wise Q4K quantization for 2D weight matrices.
 * Each row is padded to 256-element boundary before quantization.
 * Compatible with fused matmul kernel: blocks align to row boundaries.
 *
 * Layout: Row 0 blocks (ceil(K/256)), Row 1 blocks, ...
 * Total blocks: rows × ceil(cols/256)
 */
export function quantizeToQ4KMRowWise(data: Float32Array, shape: [number, number]): QuantizeResult {
  const [rows, cols] = shape;
  const numElements = rows * cols;

  if (data.length !== numElements) {
    throw new Error(`Data length ${data.length} doesn't match shape ${shape}`);
  }

  const blocksPerRow = Math.ceil(cols / QK_K);
  const paddedColsPerRow = blocksPerRow * QK_K;
  const totalBlocks = rows * blocksPerRow;

  const quantized = new Uint8Array(totalBlocks * QK4_K_BLOCK_SIZE);

  for (let row = 0; row < rows; row++) {
    // Extract and pad this row
    const rowData = new Float32Array(paddedColsPerRow);
    const srcOffset = row * cols;
    for (let c = 0; c < cols; c++) {
      rowData[c] = data[srcOffset + c];
    }
    // Remaining elements are zero-padded (Float32Array default)

    // Quantize each block in this row
    for (let b = 0; b < blocksPerRow; b++) {
      const block = quantizeQ4KBlock(rowData, b * QK_K);
      const dstOffset = (row * blocksPerRow + b) * QK4_K_BLOCK_SIZE;
      quantized.set(block, dstOffset);
    }
  }

  return {
    quantized,
    numBlocks: totalBlocks,
    originalSize: numElements * 4,
    quantizedSize: quantized.length,
    compressionRatio: (numElements * 4) / quantized.length,
  };
}

/**
 * Transpose a 2D F32 matrix.
 * W[rows, cols] → W^T[cols, rows]
 */
export function transposeF32(data: Float32Array, shape: [number, number]): Float32Array {
  const [rows, cols] = shape;
  const transposed = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      transposed[c * rows + r] = data[r * cols + c];
    }
  }
  return transposed;
}

/**
 * Column-wise Q4K quantization for 2D weight matrices.
 * Transposes W[out, K] → W^T[K, out], then applies row-wise quantization.
 *
 * Benefits:
 * 1. GPU threads read contiguous memory (coalesced access) → 1.5-2x faster
 * 2. Blocks align to columns of original matrix
 *
 * Layout (after transpose): W^T row 0 blocks, W^T row 1 blocks, ...
 * Stored shape: [K, out] (transposed)
 * Total blocks: K × ceil(out/256)
 */
export function quantizeToQ4KMColumnWise(data: Float32Array, shape: [number, number]): QuantizeResult & { transposedShape: [number, number] } {
  const [rows, cols] = shape; // Original: [out, K]

  // Transpose: W[out, K] → W^T[K, out]
  const transposed = transposeF32(data, shape);
  const transposedShape: [number, number] = [cols, rows]; // [K, out]

  // Quantize transposed matrix with row-wise packing
  const result = quantizeToQ4KMRowWise(transposed, transposedShape);

  return {
    ...result,
    transposedShape,
  };
}

export type Q4KLayout = 'flat' | 'row_wise' | 'column_wise';

/**
 * Get expected Q4K size for different layouts.
 */
export function getQ4KSize(shape: number[], layout: Q4KLayout = 'flat'): number {
  const numElements = shape.reduce((a, b) => a * b, 1);

  if (layout === 'flat' || shape.length !== 2) {
    const numBlocks = Math.ceil(numElements / QK_K);
    return numBlocks * QK4_K_BLOCK_SIZE;
  }

  const [rows, cols] = shape as [number, number];

  if (layout === 'row_wise') {
    const blocksPerRow = Math.ceil(cols / QK_K);
    return rows * blocksPerRow * QK4_K_BLOCK_SIZE;
  }

  if (layout === 'column_wise') {
    // After transpose: [cols, rows], row-wise on that
    const blocksPerRow = Math.ceil(rows / QK_K);
    return cols * blocksPerRow * QK4_K_BLOCK_SIZE;
  }

  return Math.ceil(numElements / QK_K) * QK4_K_BLOCK_SIZE;
}

export function dequantizeQ4KM(quantized: Uint8Array, numBlocks: number, shape: number[]): Float32Array {
  const numElements = shape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(numElements);

  for (let b = 0; b < numBlocks; b++) {
    const blockOffset = b * QK4_K_BLOCK_SIZE;
    const block = quantized.slice(blockOffset, blockOffset + QK4_K_BLOCK_SIZE);
    const dequantized = dequantizeQ4KBlock(block);

    const startIdx = b * QK_K;
    const copyLen = Math.min(QK_K, numElements - startIdx);
    for (let i = 0; i < copyLen; i++) {
      result[startIdx + i] = dequantized[i];
    }
  }

  return result;
}

export function calculateQuantizationError(original: Float32Array, reconstructed: Float32Array): QuantizationError {
  if (original.length !== reconstructed.length) {
    throw new Error('Length mismatch');
  }

  let mse = 0;
  let maxError = 0;
  let signalPower = 0;

  for (let i = 0; i < original.length; i++) {
    const diff = original[i] - reconstructed[i];
    mse += diff * diff;
    maxError = Math.max(maxError, Math.abs(diff));
    signalPower += original[i] * original[i];
  }

  mse /= original.length;
  signalPower /= original.length;

  const snr = signalPower > 0 ? 10 * Math.log10(signalPower / mse) : Infinity;

  return { mse, maxError, snr };
}

export function quantizeF16ToQ4KM(f16Data: Uint16Array, shape: number[]): QuantizeResult {
  const f32Data = new Float32Array(f16Data.length);
  for (let i = 0; i < f16Data.length; i++) {
    f32Data[i] = float16ToFloat32(f16Data[i]);
  }
  return quantizeToQ4KM(f32Data, shape);
}

export interface QuantizeOptions {
  /** Also quantize embedding tables (default: false) */
  quantizeEmbeddings?: boolean;
  /** Modules to skip (from HF config) */
  modulesToNotConvert?: string[] | null;
}

export function shouldQuantize(
  tensorName: string,
  shape: number[],
  options: QuantizeOptions = {}
): boolean {
  const { quantizeEmbeddings = false, modulesToNotConvert = null } = options;

  const numElements = shape.reduce((a, b) => a * b, 1);
  if (numElements < 1024) {
    return false;
  }

  const lowerName = tensorName.toLowerCase();

  // Embeddings: skip unless explicitly enabled
  if (lowerName.includes('embed') || lowerName.includes('lm_head')) {
    if (!quantizeEmbeddings) {
      return false;
    }
    // If embeddings enabled, continue to other checks
  }

  if (lowerName.includes('norm') || lowerName.includes('ln_')) {
    return false;
  }

  if (lowerName.endsWith('.bias') || lowerName.endsWith('_bias')) {
    return false;
  }

  if (lowerName.includes('router') || lowerName.includes('gate.weight')) {
    return false;
  }

  if (modulesToNotConvert && Array.isArray(modulesToNotConvert)) {
    for (const pattern of modulesToNotConvert) {
      const regexPattern = pattern
        .replace(/\./g, '\\.')
        .replace(/\*/g, '\\d+');
      const regex = new RegExp(regexPattern);
      if (regex.test(tensorName)) {
        return false;
      }
    }
  }

  return true;
}

export function getQuantizedSize(shape: number[]): number {
  const numElements = shape.reduce((a, b) => a * b, 1);
  const numBlocks = Math.ceil(numElements / QK_K);
  return numBlocks * QK4_K_BLOCK_SIZE;
}

export { QK_K, QK4_K_BLOCK_SIZE };
