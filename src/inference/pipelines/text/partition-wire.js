import { createCommandRecorder } from '../../../gpu/command-recorder.js';
import { createTensor } from '../../../gpu/tensor.js';
import { recordCastF32ToF16 } from '../../../gpu/kernels/cast.js';
import { recordCheckFinitenessF32 } from '../../../gpu/kernels/check-finiteness-f32.js';
import { acquireBuffer, releaseBuffer, discardBuffer, readBuffer, uploadData } from '../../../memory/buffer-pool.js';

/** Explicit GPU narrowing at a partition boundary, not a model-layer precision fallback.
 * Input is borrowed; the output bytes own their storage. Guard and activation share
 * one readback. The f32 guard rejects NaN, infinity and values outside finite binary16.
 * @type {import('./partition-wire.js').readPartitionF16Boundary} */
export async function readPartitionF16Boundary(input, shape, signal) {
  signal.throwIfAborted();
  if (!Array.isArray(shape) || shape.length !== 3 || shape.some(value => !Number.isSafeInteger(value) || value < 1)) {
    throw new Error('Partition wire conversion requires explicit three-dimensional geometry.');
  }
  const size = shape.reduce((product, value) => product * value, 1);
  if (!Number.isSafeInteger(size) || size > 0xffffffff || input.size < size * 4) {
    throw new Error('Partition wire conversion exceeds its input allocation.');
  }
  const byteLength = size * 2;
  const guardOffset = Math.ceil(byteLength / 4) * 4;
  const recorder = createCommandRecorder('resident_partition_wire');
  let output = null;
  let submitted = false;
  let completed = false;
  try {
    output = acquireBuffer(guardOffset + 16, undefined, 'resident_partition_wire_output');
    const status = acquireBuffer(16, undefined, 'resident_partition_wire_status');
    recorder.trackTemporaryBuffer(status);
    uploadData(status, new Uint32Array(4));
    await recordCheckFinitenessF32(recorder, input, size, status, 0, 0, 65504);
    await recordCastF32ToF16(recorder, createTensor(input, 'f32', shape, 'resident_partition_boundary'), { outputBuffer: output });
    recorder.getEncoder().copyBufferToBuffer(status, 0, output, guardOffset, 16);
    signal.throwIfAborted();
    submitted = true;
    await recorder.submitAndWait();
    completed = true;
    signal.throwIfAborted();
    const bytes = await readBuffer(output, guardOffset + 16);
    signal.throwIfAborted();
    if (new DataView(bytes).getUint32(guardOffset, true) !== 0) {
      throw new Error('Partition f16 transport rejected nonfinite or overflowing activations.');
    }
    return bytes.slice(0, byteLength);
  } finally {
    try { await recorder.abort(); } finally {
      if (output) (submitted && !completed ? discardBuffer : releaseBuffer)(output);
    }
  }
}
