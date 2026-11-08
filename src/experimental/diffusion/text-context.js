// SD3 joint text context assembly.
//
// The reference SD3 pipeline conditions the transformer on three encoders:
// the two CLIP penultimate hidden states are concatenated along the feature
// axis, zero-padded to the T5 width (joint_attention_dim, 4096), and placed in
// front of the T5 hidden states along the sequence axis. The pooled CLIP
// projections feed the timestep embedding separately. This module plans the
// byte copies for that layout and performs them on the GPU.

import { getDevice } from '../../gpu/device.js';
import { createTensor } from '../../gpu/tensor.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';

/** @typedef {import('../../gpu/tensor.js').Tensor} Tensor */
/** @typedef {{ source: number, sourceOffset: number, destOffset: number, byteLength: number }} Sd3TextContextCopy */

const COPY_ALIGNMENT_BYTES = 4;

/**
 * @param {unknown} value
 * @param {string} label
 * @returns {number}
 */
function requirePositiveInteger(value, label) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(`SD3 text context requires ${label} to be a positive integer; got ${String(value)}.`);
  }
  return value;
}

/**
 * @param {number} bytes
 * @param {string} label
 * @returns {number}
 */
function requireAligned(bytes, label) {
  if (bytes % COPY_ALIGNMENT_BYTES !== 0) {
    throw new Error(`SD3 text context ${label} (${bytes} bytes) must be a multiple of ${COPY_ALIGNMENT_BYTES} bytes for buffer copies.`);
  }
  return bytes;
}

/**
 * @param {string} dtype
 * @returns {number}
 */
export function bytesPerElementForDtype(dtype) {
  if (dtype === 'f16') return 2;
  if (dtype === 'f32') return 4;
  throw new Error(`SD3 text context supports f16 or f32 hidden states; got "${String(dtype)}".`);
}

// Pure layout plan. `clipDims` lists the CLIP encoders' hidden sizes in order;
// each contributes `clipRows` rows (its tokenizer max length). The T5 block is
// one contiguous copy after the CLIP rows.
/**
 * @param {{ clipRows: number, clipDims: number[], t5Rows: number, jointDim: number, bytesPerElement: number }} input
 * @returns {{ rows: number, jointDim: number, clipWidth: number, paddedPerClipRow: number, totalBytes: number, copies: Sd3TextContextCopy[] }}
 */
export function planSd3TextContext({ clipRows, clipDims, t5Rows, jointDim, bytesPerElement }) {
  requirePositiveInteger(clipRows, 'clipRows');
  requirePositiveInteger(t5Rows, 't5Rows');
  requirePositiveInteger(jointDim, 'jointDim');
  requirePositiveInteger(bytesPerElement, 'bytesPerElement');
  if (!Array.isArray(clipDims) || clipDims.length === 0) {
    throw new Error('SD3 text context requires at least one CLIP hidden size.');
  }
  clipDims.forEach((dim, index) => requirePositiveInteger(dim, `clipDims[${index}]`));
  const clipWidth = clipDims.reduce((sum, dim) => sum + dim, 0);
  if (clipWidth > jointDim) {
    throw new Error(`SD3 text context CLIP width ${clipWidth} exceeds the joint width ${jointDim}.`);
  }
  const rowBytes = requireAligned(jointDim * bytesPerElement, 'row stride');
  /** @type {Sd3TextContextCopy[]} */
  const copies = [];
  let featureOffset = 0;
  clipDims.forEach((dim, source) => {
    const sourceRowBytes = requireAligned(dim * bytesPerElement, `CLIP ${source} row`);
    requireAligned(featureOffset * bytesPerElement, `CLIP ${source} feature offset`);
    for (let row = 0; row < clipRows; row += 1) {
      copies.push({
        source,
        sourceOffset: row * sourceRowBytes,
        destOffset: row * rowBytes + featureOffset * bytesPerElement,
        byteLength: sourceRowBytes,
      });
    }
    featureOffset += dim;
  });
  copies.push({
    source: clipDims.length,
    sourceOffset: 0,
    destOffset: clipRows * rowBytes,
    byteLength: t5Rows * rowBytes,
  });
  const rows = clipRows + t5Rows;
  return {
    rows,
    jointDim,
    clipWidth,
    paddedPerClipRow: jointDim - clipWidth,
    totalBytes: rows * rowBytes,
    copies,
  };
}

// Assemble the context tensor on the GPU from the encoder outputs. Takes
// ownership of the input tensors' buffers and releases them.
/**
 * @param {{ clipHidden: Tensor[], t5Hidden: Tensor }} input
 * @returns {Promise<Tensor>}
 */
export async function assembleSd3TextContext({ clipHidden, t5Hidden }) {
  if (!Array.isArray(clipHidden) || clipHidden.length === 0 || !t5Hidden) {
    throw new Error('SD3 text context assembly requires CLIP hidden states and a T5 hidden state.');
  }
  const dtype = t5Hidden.dtype;
  const bytesPerElement = bytesPerElementForDtype(dtype);
  for (const [index, tensor] of clipHidden.entries()) {
    if (tensor.dtype !== dtype) {
      throw new Error(`SD3 text context CLIP ${index} hidden dtype "${tensor.dtype}" differs from T5 dtype "${dtype}".`);
    }
    if (tensor.shape[0] !== clipHidden[0].shape[0]) {
      throw new Error(`SD3 text context CLIP ${index} has ${tensor.shape[0]} rows; CLIP 0 has ${clipHidden[0].shape[0]}.`);
    }
  }
  const plan = planSd3TextContext({
    clipRows: clipHidden[0].shape[0],
    clipDims: clipHidden.map((tensor) => tensor.shape[1]),
    t5Rows: t5Hidden.shape[0],
    jointDim: t5Hidden.shape[1],
    bytesPerElement,
  });
  const sources = [...clipHidden.map((tensor) => tensor.buffer), t5Hidden.buffer];
  const device = getDevice();
  if (!device) throw new Error('SD3 text context assembly requires a WebGPU device.');
  const output = acquireBuffer(plan.totalBytes, undefined, 'sd3_text_context');
  try {
    const encoder = device.createCommandEncoder({ label: 'sd3_text_context' });
    // The pool may hand back a reused buffer; the CLIP rows' tail must be zero.
    encoder.clearBuffer(output, 0, plan.totalBytes);
    for (const copy of plan.copies) {
      encoder.copyBufferToBuffer(sources[copy.source], copy.sourceOffset, output, copy.destOffset, copy.byteLength);
    }
    device.queue.submit([encoder.finish()]);
  } catch (error) {
    releaseBuffer(output);
    throw error;
  }
  for (const buffer of sources) releaseBuffer(buffer);
  return createTensor(output, dtype, [plan.rows, plan.jointDim], 'sd3_text_context');
}
