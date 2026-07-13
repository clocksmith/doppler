import { QK_K, Q4K_BLOCK_BYTES } from '../../src/loader/quantization-constants.js';
import { float16ToFloat32 } from '../../src/converter/quantizer.js';

function assertFloatArray(value, label) {
  if (!(value instanceof Float32Array)) {
    throw new Error(`${label} must be a Float32Array.`);
  }
}

function assertMatrixShape(shape, label) {
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error(`${label} shape must be [rows, columns].`);
  }
  const [rows, columns] = shape;
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(columns) || columns <= 0) {
    throw new Error(`${label} shape must contain positive integers.`);
  }
  return [rows, columns];
}

function decodeBlockInto(packed, byteOffset, values, scales, minima, scaleBits, minBits) {
  const view = new DataView(packed.buffer, packed.byteOffset + byteOffset, Q4K_BLOCK_BYTES);
  const d = float16ToFloat32(view.getUint16(0, true));
  const dmin = float16ToFloat32(view.getUint16(2, true));

  for (let index = 0; index < 4; index += 1) {
    scaleBits[index] = packed[byteOffset + 4 + index] & 0x3f;
    scaleBits[index + 4] = ((packed[byteOffset + 4 + index] >> 6) & 0x03) << 4;
    minBits[index] = packed[byteOffset + 8 + index] & 0x3f;
    minBits[index + 4] = ((packed[byteOffset + 8 + index] >> 6) & 0x03) << 4;
  }

  for (let index = 0; index < 4; index += 1) {
    scaleBits[index + 4] |= packed[byteOffset + 12 + index] & 0x0f;
    minBits[index + 4] |= (packed[byteOffset + 12 + index] >> 4) & 0x0f;
  }

  for (let index = 0; index < 8; index += 1) {
    scales[index] = d * scaleBits[index];
    minima[index] = dmin * minBits[index];
  }

  for (let chunk = 0; chunk < 4; chunk += 1) {
    const chunkBase = chunk * 64;
    const quantBase = byteOffset + 16 + chunk * 32;
    const lowerSubblock = chunk * 2;
    const upperSubblock = lowerSubblock + 1;
    for (let index = 0; index < 32; index += 1) {
      const quant = packed[quantBase + index];
      values[chunkBase + index] = scales[lowerSubblock] * (quant & 0x0f) - minima[lowerSubblock];
      values[chunkBase + 32 + index] = scales[upperSubblock] * ((quant >> 4) & 0x0f) - minima[upperSubblock];
    }
  }

  return { d, dmin };
}

export function decodeQ4KBlockReference(packed, byteOffset = 0) {
  if (!(packed instanceof Uint8Array)) {
    throw new Error('packed must be a Uint8Array.');
  }
  if (!Number.isInteger(byteOffset) || byteOffset < 0 || byteOffset + Q4K_BLOCK_BYTES > packed.byteLength) {
    throw new Error('byteOffset does not address a complete Q4_K block.');
  }
  const values = new Float32Array(QK_K);
  const scales = new Float32Array(8);
  const minima = new Float32Array(8);
  const scaleBits = new Uint8Array(8);
  const minBits = new Uint8Array(8);
  const multipliers = decodeBlockInto(
    packed,
    byteOffset,
    values,
    scales,
    minima,
    scaleBits,
    minBits
  );
  return { values, scales, minima, scaleBits, minBits, ...multipliers };
}

export function projectQ4KRowWiseReference(packed, shape, activation, options = {}) {
  if (!(packed instanceof Uint8Array)) {
    throw new Error('packed must be a Uint8Array.');
  }
  const [rows, columns] = assertMatrixShape(shape, 'Q4_K tensor');
  assertFloatArray(activation, 'activation');
  if (activation.length !== columns) {
    throw new Error(`activation length ${activation.length} does not match ${columns} columns.`);
  }

  const blocksPerRow = Math.ceil(columns / QK_K);
  const totalBlocks = rows * blocksPerRow;
  const expectedBytes = totalBlocks * Q4K_BLOCK_BYTES;
  if (packed.byteLength !== expectedBytes) {
    throw new Error(`packed byte length ${packed.byteLength} does not match expected ${expectedBytes}.`);
  }

  const output = new Float32Array(rows);
  const decodedScales = new Float32Array(totalBlocks * 8);
  const decodedMinima = new Float32Array(totalBlocks * 8);
  const packedScaleBits = new Uint8Array(totalBlocks * 8);
  const packedMinBits = new Uint8Array(totalBlocks * 8);
  const blockD = new Float32Array(totalBlocks);
  const blockDmin = new Float32Array(totalBlocks);

  const values = new Float32Array(QK_K);
  const scales = new Float32Array(8);
  const minima = new Float32Array(8);
  const scaleBits = new Uint8Array(8);
  const minBits = new Uint8Array(8);
  const decodedRow = new Float32Array(columns);

  let decodedMinimum = Infinity;
  let decodedMaximum = -Infinity;
  let decodedSum = 0;
  let decodedSumSquares = 0;

  for (let row = 0; row < rows; row += 1) {
    let sum = 0;
    let rowOffset = 0;
    for (let blockInRow = 0; blockInRow < blocksPerRow; blockInRow += 1) {
      const blockIndex = row * blocksPerRow + blockInRow;
      const byteOffset = blockIndex * Q4K_BLOCK_BYTES;
      const { d, dmin } = decodeBlockInto(
        packed,
        byteOffset,
        values,
        scales,
        minima,
        scaleBits,
        minBits
      );
      blockD[blockIndex] = d;
      blockDmin[blockIndex] = dmin;
      decodedScales.set(scales, blockIndex * 8);
      decodedMinima.set(minima, blockIndex * 8);
      packedScaleBits.set(scaleBits, blockIndex * 8);
      packedMinBits.set(minBits, blockIndex * 8);

      const valueCount = Math.min(QK_K, columns - rowOffset);
      for (let index = 0; index < valueCount; index += 1) {
        const value = values[index];
        decodedRow[rowOffset + index] = value;
        sum += activation[rowOffset + index] * value;
        decodedMinimum = Math.min(decodedMinimum, value);
        decodedMaximum = Math.max(decodedMaximum, value);
        decodedSum += value;
        decodedSumSquares += value * value;
      }
      rowOffset += valueCount;
    }
    output[row] = sum;
    options.onDecodedRow?.(row, decodedRow);
  }

  const elementCount = rows * columns;
  const decodedMean = decodedSum / elementCount;
  return {
    output,
    decodedScales,
    decodedMinima,
    packedScaleBits,
    packedMinBits,
    blockD,
    blockDmin,
    decodedWeightStats: {
      elementCount,
      min: decodedMinimum,
      max: decodedMaximum,
      mean: decodedMean,
      std: Math.sqrt(Math.max(0, decodedSumSquares / elementCount - decodedMean * decodedMean)),
    },
  };
}

export function projectF16RowWiseReference(packed, shape, activation) {
  if (!(packed instanceof Uint8Array)) {
    throw new Error('packed must be a Uint8Array.');
  }
  const [rows, columns] = assertMatrixShape(shape, 'F16 tensor');
  assertFloatArray(activation, 'activation');
  if (activation.length !== columns) {
    throw new Error(`activation length ${activation.length} does not match ${columns} columns.`);
  }
  const expectedBytes = rows * columns * 2;
  if (packed.byteLength !== expectedBytes) {
    throw new Error(`packed byte length ${packed.byteLength} does not match expected ${expectedBytes}.`);
  }

  const output = new Float32Array(rows);
  const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  for (let row = 0; row < rows; row += 1) {
    let sum = 0;
    const rowOffset = row * columns;
    for (let column = 0; column < columns; column += 1) {
      const weight = float16ToFloat32(view.getUint16((rowOffset + column) * 2, true));
      sum += activation[column] * weight;
    }
    output[row] = sum;
  }
  return output;
}

export function compareFloatArrays(left, right) {
  assertFloatArray(left, 'left');
  assertFloatArray(right, 'right');
  if (left.length !== right.length) {
    throw new Error(`array length mismatch: ${left.length} versus ${right.length}.`);
  }

  let maxAbsDiff = 0;
  let maxRelDiff = 0;
  let worstIndex = -1;
  let sumAbs = 0;
  let sumSquares = 0;
  let dot = 0;
  let leftSquares = 0;
  let rightSquares = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index];
    const rightValue = right[index];
    const difference = Math.abs(leftValue - rightValue);
    const relative = difference / Math.max(Math.abs(leftValue), Math.abs(rightValue), 1e-12);
    if (difference > maxAbsDiff) {
      maxAbsDiff = difference;
      worstIndex = index;
    }
    maxRelDiff = Math.max(maxRelDiff, relative);
    sumAbs += difference;
    sumSquares += difference * difference;
    dot += leftValue * rightValue;
    leftSquares += leftValue * leftValue;
    rightSquares += rightValue * rightValue;
  }

  return {
    elementCount: left.length,
    maxAbsDiff,
    maxRelDiff,
    meanAbsDiff: sumAbs / left.length,
    rmse: Math.sqrt(sumSquares / left.length),
    worstIndex,
    leftAtWorst: worstIndex >= 0 ? left[worstIndex] : null,
    rightAtWorst: worstIndex >= 0 ? right[worstIndex] : null,
    cosineSimilarity: dot / Math.max(Math.sqrt(leftSquares * rightSquares), 1e-30),
  };
}
