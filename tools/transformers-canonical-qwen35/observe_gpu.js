// Observation only: copy actual GPU bytes without downloading or replacing the
// inference tensor. Consumers may hash locally or retain diagnostic samples.
export async function observationDeadline(operation, timeoutMs, name) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) throw new Error('GPU observation requires an explicit per-operation timeout');
  let timer;
  try { return await Promise.race([operation, new Promise((resolve, reject) => {
    timer = setTimeout(() => reject(new Error(`GPU observation timed out at ${name}`)), timeoutMs);
  })]); } finally { clearTimeout(timer); }
}

async function visitGpuBytes(device, tensor, bytes, chunkBytes, timeoutMs, consume) {
  if (tensor.location !== 'gpu-buffer' || tensor.type !== 'float32') throw new Error('GPU observation requires an actual F32 GPU buffer');
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) throw new Error('GPU observation requires an explicit per-operation timeout');
  const buffer = device.createBuffer({ size: Math.min(chunkBytes, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    for (let offset = 0; offset < bytes; offset += chunkBytes) {
      const count = Math.min(chunkBytes, bytes - offset);
      device.pushErrorScope('validation');
      const command = device.createCommandEncoder();
      command.copyBufferToBuffer(tensor.gpuBuffer, offset, buffer, 0, count);
      device.queue.submit([command.finish()]);
      const validationError = await observationDeadline(device.popErrorScope(), timeoutMs, 'popErrorScope');
      if (validationError) throw new Error('GPU observation copy failed: ' + validationError.message);
      await observationDeadline(buffer.mapAsync(GPUMapMode.READ, 0, count), timeoutMs, 'mapAsync');
      try {
        await consume(new Uint8Array(buffer.getMappedRange(0, count)), offset);
      } finally { buffer.unmap(); }
    }
  } finally { buffer.destroy(); }
}

export async function readGpuTensor(device, tensor, bytes, chunkBytes, timeoutMs, maxAllocationBytes) {
  if (!Number.isSafeInteger(maxAllocationBytes) || bytes > maxAllocationBytes || bytes < 4 || bytes % 4) {
    throw new Error('GPU observation exceeds its explicit CPU allocation bound');
  }
  const observed = new Uint8Array(bytes);
  await visitGpuBytes(device, tensor, bytes, chunkBytes, timeoutMs, (chunk, offset) => observed.set(chunk, offset));
  return observed;
}

export async function uploadGpuTensor(device, tensor, bytes, chunkBytes, url, timeoutMs) {
  let last;
  await visitGpuBytes(device, tensor, bytes, chunkBytes, timeoutMs, async (chunk, offset) => {
    const separator = url.includes('?') ? '&' : '?';
    const response = await fetch(`${url}${separator}offset=${offset}`, { method: 'POST', body: chunk,
      signal: AbortSignal.timeout(timeoutMs) });
    if (!response.ok) throw new Error('Host GPU-observation range was rejected: ' + await response.text());
    last = await response.json();
  });
  if (!last?.receipt || last.bytes !== bytes) throw new Error('GPU observation did not complete every byte');
  return last.receipt;
}
