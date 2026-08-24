import { f16ToF32 } from '../../../../loader/dtype-utils.js';

const bf16ScratchU32 = new Uint32Array(1);
const bf16ScratchF32 = new Float32Array(bf16ScratchU32.buffer);
const SPLIT_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024;

function bf16ToF32(value) {
  bf16ScratchU32[0] = (value & 0xffff) << 16;
  return bf16ScratchF32[0];
}

export function isRangeBackedCpuWeightSource(value) {
  return (
    typeof value === 'object'
    && value !== null
    && value.kind === 'tensor_range_source'
    && typeof value.loadRange === 'function'
  );
}

export function normalizeRangeBytes(value, label) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new Error(
    `[Logits] ${label} returned unsupported byte payload type "${value?.constructor?.name ?? typeof value}".`
  );
}

function decodeChunkIntoOutput(bytes, sourceDtype, output, dstOffset, valueCount) {
  if (sourceDtype === 'f16') {
    const values = new Uint16Array(bytes.buffer, bytes.byteOffset, valueCount);
    for (let index = 0; index < valueCount; index += 1) {
      output[dstOffset + index] = f16ToF32(values[index]);
    }
    return;
  }
  if (sourceDtype === 'bf16') {
    const values = new Uint16Array(bytes.buffer, bytes.byteOffset, valueCount);
    for (let index = 0; index < valueCount; index += 1) {
      output[dstOffset + index] = bf16ToF32(values[index]);
    }
    return;
  }
  if (((bytes.byteOffset % 4) === 0) && ((bytes.byteLength % 4) === 0)) {
    output.set(new Float32Array(bytes.buffer, bytes.byteOffset, valueCount), dstOffset);
    return;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < valueCount; index += 1) {
    output[dstOffset + index] = view.getFloat32(index * 4, true);
  }
}

function readTypedLmHeadChunk(data, layout, hiddenSize, vocabSize, rowOffset, rowCount, sourceDtype) {
  if (data instanceof Float32Array) {
    if (layout === 'row') {
      const start = rowOffset * hiddenSize;
      return data.subarray(start, start + rowCount * hiddenSize);
    }
    const chunk = new Float32Array(hiddenSize * rowCount);
    for (let k = 0; k < hiddenSize; k++) {
      const srcOffset = k * vocabSize + rowOffset;
      const dstOffset = k * rowCount;
      chunk.set(data.subarray(srcOffset, srcOffset + rowCount), dstOffset);
    }
    return chunk;
  }

  if (!(data instanceof Uint16Array)) {
    throw new Error(
      `[Logits] Unsupported CPU LM head chunk source type "${data?.constructor?.name ?? typeof data}".`
    );
  }

  const chunk = new Float32Array(hiddenSize * rowCount);
  if (layout === 'row') {
    const start = rowOffset * hiddenSize;
    for (let index = 0; index < rowCount * hiddenSize; index += 1) {
      const raw = data[start + index];
      chunk[index] = sourceDtype === 'bf16' ? bf16ToF32(raw) : f16ToF32(raw);
    }
    return chunk;
  }

  for (let k = 0; k < hiddenSize; k += 1) {
    const srcOffset = k * vocabSize + rowOffset;
    const dstOffset = k * rowCount;
    for (let index = 0; index < rowCount; index += 1) {
      const raw = data[srcOffset + index];
      chunk[dstOffset + index] = sourceDtype === 'bf16' ? bf16ToF32(raw) : f16ToF32(raw);
    }
  }
  return chunk;
}


export function resolveCpuWeightDims(lmHead) {
  if (lmHead.shape.length !== 2) {
    throw new Error(`[Logits] CPU LM head shape must be 2D, got [${lmHead.shape.join(', ')}]`);
  }
  if (lmHead.layout === 'column') {
    return { hiddenSize: lmHead.shape[0], vocabSize: lmHead.shape[1] };
  }
  return { vocabSize: lmHead.shape[0], hiddenSize: lmHead.shape[1] };
}


export async function extractLmHeadChunk(
  data,
  layout,
  hiddenSize,
  vocabSize,
  rowOffset,
  rowCount,
  sourceDtype
) {
  if (typeof sourceDtype !== 'string' || sourceDtype.trim() === '') {
    throw new Error('[Logits] CPU LM head source dtype is required.');
  }
  const normalizedSourceDtype = sourceDtype.trim().toLowerCase();
  if (normalizedSourceDtype !== 'f32' && normalizedSourceDtype !== 'f16' && normalizedSourceDtype !== 'bf16') {
    throw new Error(`[Logits] Unsupported CPU LM head source dtype "${sourceDtype}".`);
  }
  if (!isRangeBackedCpuWeightSource(data)) {
    return readTypedLmHeadChunk(
      data,
      layout,
      hiddenSize,
      vocabSize,
      rowOffset,
      rowCount,
      normalizedSourceDtype
    );
  }

  const bytesPerElement = normalizedSourceDtype === 'f32' ? 4 : 2;
  const chunk = new Float32Array(hiddenSize * rowCount);
  if (layout === 'row') {
    const byteOffset = rowOffset * hiddenSize * bytesPerElement;
    const byteLength = rowCount * hiddenSize * bytesPerElement;
    const bytes = normalizeRangeBytes(
      await data.loadRange(byteOffset, byteLength),
      'CPU LM head range source'
    );
    if (bytes.byteLength !== byteLength) {
      throw new Error(
        `[Logits] CPU LM head range source returned ${bytes.byteLength} bytes, expected ${byteLength}.`
      );
    }
    decodeChunkIntoOutput(bytes, normalizedSourceDtype, chunk, 0, rowCount * hiddenSize);
    return chunk;
  }

  for (let k = 0; k < hiddenSize; k += 1) {
    const byteOffset = (k * vocabSize + rowOffset) * bytesPerElement;
    const byteLength = rowCount * bytesPerElement;
    const bytes = normalizeRangeBytes(
      await data.loadRange(byteOffset, byteLength),
      `CPU LM head range source column ${k}`
    );
    if (bytes.byteLength !== byteLength) {
      throw new Error(
        `[Logits] CPU LM head range source returned ${bytes.byteLength} bytes for column ${k}, expected ${byteLength}.`
      );
    }
    decodeChunkIntoOutput(bytes, normalizedSourceDtype, chunk, k * rowCount, rowCount);
  }
  return chunk;
}


export function writeChunkLogits(
  target,
  chunk,
  numTokens,
  vocabSize,
  rowOffset,
  rowCount
) {
  for (let t = 0; t < numTokens; t++) {
    const srcOffset = t * rowCount;
    const dstOffset = t * vocabSize + rowOffset;
    target.set(chunk.subarray(srcOffset, srcOffset + rowCount), dstOffset);
  }
}

export function shouldMaterializeSplitLmHeadGPU(lmHead, largeWeightConfig) {
  const overrides = largeWeightConfig?.gpuResidentOverrides;
  if (!Array.isArray(overrides) || overrides.length === 0) {
    return false;
  }
  const label = lmHead?.label;
  return typeof label === 'string' && overrides.includes(label);
}

