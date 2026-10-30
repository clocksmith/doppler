// Observation only: copy actual GPU bytes without downloading or replacing the
// inference tensor. The host incrementally hashes the streamed byte ranges.
export async function uploadGpuTensor(device, tensor, bytes, chunkBytes, url, timeoutMs) {
  if (tensor.location !== 'gpu-buffer' || tensor.type !== 'float32') throw new Error('GPU observation requires an actual F32 GPU buffer');
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) throw new Error('GPU observation requires an explicit per-operation timeout');
  const bounded = async (operation, name) => {
    let timer;
    try { return await Promise.race([operation, new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error(`GPU observation timed out at ${name}`)), timeoutMs);
    })]); } finally { clearTimeout(timer); }
  };
  const buffer = device.createBuffer({ size: Math.min(chunkBytes, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    let last;
    for (let offset = 0; offset < bytes; offset += chunkBytes) {
      const count = Math.min(chunkBytes, bytes - offset);
      device.pushErrorScope('validation');
      const command = device.createCommandEncoder();
      command.copyBufferToBuffer(tensor.gpuBuffer, offset, buffer, 0, count);
      device.queue.submit([command.finish()]);
      const validationError = await bounded(device.popErrorScope(), 'popErrorScope');
      if (validationError) throw new Error('GPU observation copy failed: ' + validationError.message);
      await bounded(buffer.mapAsync(GPUMapMode.READ, 0, count), 'mapAsync');
      try {
        const separator = url.includes('?') ? '&' : '?';
        const response = await fetch(`${url}${separator}offset=${offset}`, { method: 'POST', body: buffer.getMappedRange(0, count),
          signal: AbortSignal.timeout(timeoutMs) });
        if (!response.ok) throw new Error('Host GPU-observation range was rejected: ' + await response.text());
        last = await response.json();
      } finally { buffer.unmap(); }
    }
    if (!last?.receipt || last.bytes !== bytes) throw new Error('GPU observation did not complete every byte');
    return last.receipt;
  } finally { buffer.destroy(); }
}
