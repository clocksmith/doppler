import { acquireBuffer, releaseBuffer, uploadData } from '../memory/buffer-pool.js';
import { createWeightBuffer } from '../gpu/weight-buffer.js';
import { QK_K, Q4K_BLOCK_BYTES } from '../config/schema/quantization.schema.js';
import { dequantizeQ4KM, dequantizeQ4KMRowWise } from '../formats/q4k-dequant.js';

/** @param {string} dtype */
export function isQ4KLocationDtype(dtype) {
  return dtype === 'Q4_K_M' || dtype === 'Q4_K';
}

/**
 * @param {unknown} data
 * @returns {Uint8Array | null}
 */
export function toUint8Array(data) {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  return null;
}

// Acquire, upload, adopt as one transaction: a failed upload releases the
// buffer before rethrowing, and only an uploaded buffer joins the owner set.
/**
 * @param {ArrayBufferView} data
 * @param {string} label
 * @param {{ add?: (buffer: GPUBuffer) => unknown } | null | undefined} owner
 * @returns {GPUBuffer}
 */
export function acquireUploadedBuffer(data, label, owner = null) {
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  try {
    uploadData(buffer, data);
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
  owner?.add?.(buffer);
  return buffer;
}

// Dequantizes a 2D Q4K tensor location on the CPU and returns an f32 weight
// buffer owned by the loader context. Returns null when the location is not a
// 2D Q4K tensor or its bytes are unavailable. The caller supplies the weight
// layout it resolved for this tensor, as a value or a resolver called only
// once the bytes are in hand.
/**
 * @typedef {import('../gpu/weight-buffer.js').WeightLayout} WeightLayout
 * @param {{ loadTensor: (name: string, cache: boolean, raw: boolean) => Promise<unknown>, gpuBuffers?: { add?: (buffer: GPUBuffer) => unknown } }} ctx
 * @param {{ name: string, location: { dtype?: unknown, shape?: unknown, size: number } | null | undefined, layout: WeightLayout | (() => WeightLayout), label?: string }} request
 */
export async function loadReferenceQ4KWeight(ctx, { name, location, layout, label = `${name}_f32_reference` }) {
  if (!location || !isQ4KLocationDtype(String(location.dtype ?? '').toUpperCase())) {
    return null;
  }
  const shape = Array.isArray(location.shape) ? location.shape : null;
  if (!shape || shape.length !== 2) {
    return null;
  }

  const quantizedBytes = toUint8Array(await ctx.loadTensor(name, false, true));
  if (!quantizedBytes) {
    return null;
  }

  const rows = Number(shape[0]);
  const cols = Number(shape[1]);
  if (!Number.isFinite(rows) || rows <= 0 || !Number.isFinite(cols) || cols <= 0) {
    return null;
  }

  const f32Weights = cols % QK_K === 0
    ? dequantizeQ4KM(quantizedBytes, Math.ceil(location.size / Q4K_BLOCK_BYTES), shape)
    : dequantizeQ4KMRowWise(quantizedBytes, shape);
  const resolvedLayout = typeof layout === 'function' ? layout() : layout;
  const buffer = acquireUploadedBuffer(f32Weights, label, ctx.gpuBuffers);
  return createWeightBuffer(buffer, 'f32', resolvedLayout, shape, name);
}
