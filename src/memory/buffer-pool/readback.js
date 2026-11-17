import { allowReadback } from '../../debug/perf-guards.js';

const RESOLVED_GPU_MAP_MODE = (
  typeof GPUMapMode === 'object'
  && GPUMapMode
)
  ? GPUMapMode
  : {
    READ: 1 << 0,
    WRITE: 1 << 1,
  };

/** Maps a bounded owned staging buffer after one ordered copy submission.
 * @type {typeof import('./readback.js').readPooledBufferSlice} */
export async function readPooledBufferSlice(buffer, offset, size, recorder, owner) {
  if (!allowReadback('BufferPool.readBuffer')) {
    return new ArrayBuffer(0);
  }

  const device = owner.device;
  if (!device) {
    throw new Error('Device not initialized');
  }
  if (recorder && (recorder.device !== device || typeof recorder.getEncoder !== 'function'
    || typeof recorder.submitPartial !== 'function')) {
    throw new Error('Recorded readback requires the owning device recorder.');
  }

  if (!Number.isInteger(offset) || offset < 0) {
    throw new Error(`Invalid read offset: ${offset}`);
  }
  if (!Number.isInteger(size) || size < 0) {
    throw new Error(`Invalid read size: ${size}`);
  }
  if (offset > buffer.size) {
    throw new Error(`Read offset ${offset} exceeds buffer size ${buffer.size}`);
  }
  if (offset + size > buffer.size) {
    throw new Error(
      `Read range [${offset}, ${offset + size}) exceeds buffer size ${buffer.size}`
    );
  }
  if (offset % 4 !== 0) {
    throw new Error(`Read offset must be 4-byte aligned, got ${offset}`);
  }
  if (size === 0) {
    return new ArrayBuffer(0);
  }

  const alignedSize = Math.ceil(size / 4) * 4;
  const staging = owner.createStagingBuffer(alignedSize);
  let mapped = false;

  try {
    // Routing consumers must wait for these bytes either way. Append the copy
    // to the producing recorder instead of submitting it and then a new copy.
    const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder({ label: 'readback_encoder' });
    encoder.copyBufferToBuffer(buffer, offset, staging, 0, alignedSize);
    if (recorder) recorder.submitPartial();
    else device.queue.submit([encoder.finish()]);

    // Map and read
    await staging.mapAsync(RESOLVED_GPU_MAP_MODE.READ);
    mapped = true;
    return staging.getMappedRange(0, alignedSize).slice(0, size);
  } catch (error) {
    if (owner.isActive(staging)) {
      owner.release(staging, false);
    }
    throw error;
  } finally {
    if (mapped && owner.isActive(staging)) {
      try {
        staging.unmap();
      } catch (error) {
        owner.release(staging, false);
        throw error;
      }
      owner.release(staging, true);
    }
  }
}
