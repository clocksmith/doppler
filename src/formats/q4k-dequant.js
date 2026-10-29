import { QK_K, QK4_K_BLOCK_SIZE } from '../config/schema/quantization.schema.js';
import { float16ToFloat32 } from './float16.js';

// Q4_K_M block layout decoding (256 elements per 144-byte block).
/** @param {Uint8Array} block */
function dequantizeQ4KBlock(block) {
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

/** @type {import('./q4k-dequant.js').dequantizeQ4KM} */
export function dequantizeQ4KM(quantized, numBlocks, shape) {
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

/** @type {import('./q4k-dequant.js').dequantizeQ4KMRowWise} */
export function dequantizeQ4KMRowWise(quantized, shape) {
  if (!Array.isArray(shape) || shape.length < 2) {
    throw new Error(`Row-wise Q4K dequantization requires a matrix-like shape, got ${shape}`);
  }
  const cols = shape[shape.length - 1];
  const rows = shape.slice(0, -1).reduce((a, b) => a * b, 1);
  const blocksPerRow = Math.ceil(cols / QK_K);
  const result = new Float32Array(rows * cols);

  for (let row = 0; row < rows; row++) {
    const rowOffset = row * blocksPerRow * QK4_K_BLOCK_SIZE;
    const rowBytes = quantized.slice(rowOffset, rowOffset + (blocksPerRow * QK4_K_BLOCK_SIZE));
    const rowDequantized = dequantizeQ4KM(rowBytes, blocksPerRow, [1, cols]);
    result.set(rowDequantized, row * cols);
  }

  return result;
}
