import { getDevice } from '../../../gpu/device.js';
import { releaseBuffer, isBufferActive } from '../../../memory/buffer-pool.js';

export function createDiffusionBufferReleaser(recorder) {
  if (!recorder) {
    return (buffer) => {
      if (!buffer || !isBufferActive(buffer)) return;
      releaseBuffer(buffer);
    };
  }
  return (buffer) => {
    if (!buffer) return;
    recorder.trackTemporaryBuffer(buffer);
  };
}

export function createDiffusionBufferDestroyer(recorder) {
  if (!recorder) {
    return (buffer) => {
      if (!buffer) return;
      const device = getDevice();
      if (!device) {
        buffer.destroy();
        return;
      }
      device.queue.onSubmittedWorkDone()
        .then(() => {
          buffer.destroy();
        })
        .catch(() => {
          buffer.destroy();
        });
    };
  }
  return (buffer) => {
    if (!buffer) return;
    recorder.trackTemporaryBuffer(buffer);
  };
}

export function createDiffusionIndexBuffer(device, indices, label) {
  const buffer = device.createBuffer({
    label,
    size: indices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  try {
    device.queue.writeBuffer(buffer, 0, indices);
    return buffer;
  } catch (error) {
    buffer.destroy();
    throw error;
  }
}
