

import { classifyTensorRole } from '../storage/rdrr-format.js';
import { shouldQuantize } from '../converter/core.js';
import {
  float16ToFloat32,
  float32ToFloat16,
  quantizeQ4KBlock,
  quantizeToQ4KMColumnWise,
  getQ4KSize,
  QK_K,
  QK4_K_BLOCK_SIZE,
} from '../converter/quantizer.js';
import {
  buildQuantizationInfo,
  normalizeQuantTag,
  resolveManifestQuantization,
  resolveModelId,
  toWebGPUDtype,
} from '../converter/node-converter/quantization.js';

export {
  buildQuantizationInfo,
  normalizeQuantTag,
  resolveManifestQuantization,
  resolveModelId,
  toWebGPUDtype,
};

const BF16_VIEW = new DataView(new ArrayBuffer(4));

function bf16ToFloat32(value) {
  BF16_VIEW.setUint32(0, value << 16, true);
  return BF16_VIEW.getFloat32(0, true);
}

function concatBytes(a, b) {
  if (!a || a.length === 0) return b;
  if (!b || b.length === 0) return a;
  const combined = new Uint8Array(a.length + b.length);
  combined.set(a, 0);
  combined.set(b, a.length);
  return combined;
}

async function* decodeFloat32Chunks(chunks, dtype) {
  const upper = String(dtype || '').toUpperCase();
  const bytesPerElement = upper === 'F32' ? 4 : 2;
  let carry = new Uint8Array(0);

  for await (const chunk of chunks) {
    const merged = concatBytes(carry, chunk);
    const count = Math.floor(merged.length / bytesPerElement);
    const used = count * bytesPerElement;
    if (count > 0) {
      const view = new DataView(merged.buffer, merged.byteOffset, used);
      const out = new Float32Array(count);
      if (upper === 'F32') {
        for (let i = 0; i < count; i++) {
          out[i] = view.getFloat32(i * 4, true);
        }
      } else if (upper === 'BF16') {
        for (let i = 0; i < count; i++) {
          const bits = view.getUint16(i * 2, true);
          out[i] = bf16ToFloat32(bits);
        }
      } else {
        for (let i = 0; i < count; i++) {
          const bits = view.getUint16(i * 2, true);
          out[i] = float16ToFloat32(bits);
        }
      }
      yield out;
    }
    carry = merged.slice(used);
  }

  if (carry.length > 0) {
    throw new Error('Unaligned tensor chunk for float decoding');
  }
}

function isGptOssPackedExpertTensor(name) {
  const lower = name.toLowerCase();
  if (!lower.includes('mlp.experts.')) return false;
  return lower.includes('gate_up_proj_blocks') ||
    lower.includes('gate_up_proj_scales') ||
    lower.includes('down_proj_blocks') ||
    lower.includes('down_proj_scales');
}

export function isMatmulWeight(name, shape) {
  if (!Array.isArray(shape) || shape.length !== 2) return false;

  const matmulPatterns = [
    /\.weight$/,
    /q_proj|k_proj|v_proj|o_proj/,
    /gate_proj|up_proj|down_proj/,
    /gate\.weight|up\.weight|down\.weight/,
    /w1\.weight|w2\.weight|w3\.weight/,
    /lm_head/,
    /embed_tokens/,
  ];

  const excludePatterns = [
    /norm|layernorm|rmsnorm/i,
    /bias$/,
    /rotary|rope/i,
  ];

  for (const pattern of excludePatterns) {
    if (pattern.test(name)) return false;
  }

  for (const pattern of matmulPatterns) {
    if (pattern.test(name)) return true;
  }

  return false;
}

export function resolveTensorDtype(name, shape, origDtype, quantizationInfo) {
  const sourceDtype = origDtype === 'BF16' ? 'F16' : origDtype;
  if (!quantizationInfo) {
    return sourceDtype;
  }

  if (isGptOssPackedExpertTensor(name)) {
    return origDtype;
  }

  const role = classifyTensorRole(name);
  if (role === 'embedding') {
    return toWebGPUDtype(quantizationInfo.embeddings);
  }
  if (role === 'lm_head') {
    const headQuant = quantizationInfo.lmHead ?? quantizationInfo.embeddings;
    return toWebGPUDtype(headQuant);
  }
  if (shouldQuantize(name, shape)) {
    return toWebGPUDtype(quantizationInfo.weights);
  }

  return sourceDtype;
}

export function resolveQ4KLayout(name, shape, quantizationInfo) {
  if (!quantizationInfo?.layout) return null;
  if (!isMatmulWeight(name, shape)) return null;
  return quantizationInfo.layout;
}

export function getQ4KOutputSize(shape, layout) {
  return getQ4KSize(shape, layout ?? 'flat');
}

export function decodeTensorToFloat32(buffer, sourceDtype) {
  const upper = String(sourceDtype || '').toUpperCase();
  if (upper === 'F32') {
    return new Float32Array(buffer);
  }

  const bytes = new Uint8Array(buffer);
  if (upper === 'BF16') {
    const values = new Float32Array(bytes.length / 2);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < values.length; i++) {
      values[i] = bf16ToFloat32(view.getUint16(i * 2, true));
    }
    return values;
  }

  const values = new Float32Array(bytes.length / 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < values.length; i++) {
    values[i] = float16ToFloat32(view.getUint16(i * 2, true));
  }
  return values;
}

export async function* createQ4KChunkStream(chunks, sourceDtype, shape, layout, chunkSizeBytes) {
  if (layout === 'col') {
    throw new Error('Column-wise Q4_K_M quantization is not supported in streaming mode.');
  }
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const rowLayout = layout === 'row' && shape.length === 2;
  const cols = rowLayout ? shape[1] : 0;

  const block = new Float32Array(QK_K);
  let blockPos = 0;
  let rowRemaining = cols;
  let processed = 0;

  const targetChunkSize = Math.max(chunkSizeBytes || QK4_K_BLOCK_SIZE, QK4_K_BLOCK_SIZE);
  let outBuffer = new Uint8Array(targetChunkSize);
  let outOffset = 0;

  for await (const values of decodeFloat32Chunks(chunks, sourceDtype)) {
    for (let i = 0; i < values.length && processed < totalElements; i++) {
      block[blockPos++] = values[i];
      processed++;
      if (rowLayout) {
        rowRemaining -= 1;
      }

      const rowBoundary = rowLayout && rowRemaining === 0;
      if (blockPos === QK_K || rowBoundary) {
        if (blockPos < QK_K) {
          block.fill(0, blockPos);
        }

        const q4Block = quantizeQ4KBlock(block, 0);
        if (q4Block.byteLength > outBuffer.byteLength) {
          if (outOffset > 0) {
            yield outBuffer.subarray(0, outOffset);
            outOffset = 0;
          }
          yield q4Block;
        } else {
          if (outOffset + q4Block.byteLength > outBuffer.byteLength) {
            yield outBuffer.subarray(0, outOffset);
            outOffset = 0;
          }
          outBuffer.set(q4Block, outOffset);
          outOffset += q4Block.byteLength;
        }

        blockPos = 0;
        if (rowBoundary) {
          rowRemaining = cols;
        }
      }
    }
  }

  if (processed < totalElements) {
    throw new Error('Quantization stream ended early');
  }

  if (!rowLayout && blockPos > 0) {
    block.fill(0, blockPos);
    const q4Block = quantizeQ4KBlock(block, 0);
    if (outOffset + q4Block.byteLength > outBuffer.byteLength) {
      yield outBuffer.subarray(0, outOffset);
      outOffset = 0;
    }
    outBuffer.set(q4Block, outOffset);
    outOffset += q4Block.byteLength;
  }

  if (outOffset > 0) {
    yield outBuffer.subarray(0, outOffset);
  }
}

export async function* createF16ChunkStream(chunks, sourceDtype) {
  const upper = String(sourceDtype || '').toUpperCase();
  if (upper === 'F16') {
    for await (const chunk of chunks) {
      yield chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
    }
    return;
  }

  const bytesPerElement = upper === 'F32' ? 4 : 2;
  let carry = new Uint8Array(0);

  for await (const chunk of chunks) {
    const merged = concatBytes(carry, chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk));
    const count = Math.floor(merged.length / bytesPerElement);
    const used = count * bytesPerElement;
    if (count > 0) {
      const view = new DataView(merged.buffer, merged.byteOffset, used);
      const out = new Uint16Array(count);
      if (upper === 'F32') {
        for (let i = 0; i < count; i++) {
          out[i] = float32ToFloat16(view.getFloat32(i * 4, true));
        }
      } else if (upper === 'BF16') {
        for (let i = 0; i < count; i++) {
          out[i] = float32ToFloat16(bf16ToFloat32(view.getUint16(i * 2, true)));
        }
      } else {
        for (let i = 0; i < count; i++) {
          out[i] = float32ToFloat16(float16ToFloat32(view.getUint16(i * 2, true)));
        }
      }
      yield new Uint8Array(out.buffer);
    }
    carry = merged.slice(used);
  }

  if (carry.length > 0) {
    throw new Error('Unaligned tensor chunk for F16 conversion');
  }
}

export async function quantizeToQ4KColumnWise(data, shape) {
  return quantizeToQ4KMColumnWise(data, shape);
}
