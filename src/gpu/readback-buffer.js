// A device-injected readback for Pack runtimes. The global BufferPool is bound
// to legacy device initialization and cannot own an independently injected port.
export async function readbackBuffer(device, buffer, sizeBytes) {
  if (!Number.isSafeInteger(sizeBytes) || sizeBytes < 4 || sizeBytes % 4 !== 0 || sizeBytes > buffer.size) {
    throw new Error('Readback requires a positive aligned range within its GPU buffer.');
  }
  const staging = device.createBuffer({ label: 'doppler-pack:readback', size: sizeBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, sizeBytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    mapped = true;
    return staging.getMappedRange(0, sizeBytes).slice(0);
  } finally {
    try { if (mapped) staging.unmap(); } finally { staging.destroy(); }
  }
}
