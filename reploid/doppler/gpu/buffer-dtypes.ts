/**
 * Lightweight dtype and layout tracking for GPU buffers.
 * WebGPU buffers are untyped; this module stores per-buffer dtype/layout metadata.
 */

export type BufferDType = 'f16' | 'f32' | 'bf16' | 'q4' | 'q4k' | 'q8' | 'i32' | 'u32' | 'unknown';
export type BufferLayout = 'row' | 'column';

const dtypeMap = new WeakMap<GPUBuffer, BufferDType>();
const layoutMap = new WeakMap<GPUBuffer, BufferLayout>();

export function setBufferDtype(buffer: GPUBuffer, dtype: BufferDType): void {
  if (buffer) dtypeMap.set(buffer, dtype);
}

export function getBufferDtype(buffer: GPUBuffer): BufferDType | null {
  return buffer ? (dtypeMap.get(buffer) ?? null) : null;
}

/**
 * Set buffer layout metadata (for matmul weights).
 * 'column' = pre-transposed, use transposeB=false
 * 'row' = standard layout, use transposeB=true
 */
export function setBufferLayout(buffer: GPUBuffer, layout: BufferLayout): void {
  if (buffer) layoutMap.set(buffer, layout);
}

/**
 * Get buffer layout metadata.
 * Returns null if not set (defaults to 'row' in matmul).
 */
export function getBufferLayout(buffer: GPUBuffer): BufferLayout | null {
  return buffer ? (layoutMap.get(buffer) ?? null) : null;
}

/**
 * Check if buffer is stored in column-major (pre-transposed) format.
 */
export function isColumnMajorBuffer(buffer: GPUBuffer): boolean {
  return getBufferLayout(buffer) === 'column';
}

export function isF16Buffer(buffer: GPUBuffer): boolean {
  return getBufferDtype(buffer) === 'f16';
}

export function isBF16Buffer(buffer: GPUBuffer): boolean {
  return getBufferDtype(buffer) === 'bf16';
}

export function isQuantizedBuffer(buffer: GPUBuffer): boolean {
  const dtype = getBufferDtype(buffer);
  return dtype === 'q4' || dtype === 'q8' || dtype === 'q4k';
}

export function isQ4KBuffer(buffer: GPUBuffer): boolean {
  return getBufferDtype(buffer) === 'q4k';
}
