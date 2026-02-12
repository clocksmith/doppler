


export function createBuffer(device, data, usage = 'read') {
  let gpuUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

  if (usage === 'read' || usage === 'readwrite') {
    gpuUsage |= GPUBufferUsage.COPY_DST;
  }

  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: gpuUsage,
    mappedAtCreation: true,
  });

  // Copy data
  const arrayType = data.constructor;
  new arrayType(buffer.getMappedRange()).set(data);
  buffer.unmap();

  return buffer;
}


export function createEmptyBuffer(device, size, usage = 'readwrite') {
  let gpuUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  return device.createBuffer({
    size,
    usage: gpuUsage,
  });
}


export async function readGPUBuffer(device, buffer, size) {
  // Create staging buffer for readback
  const stagingBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Copy from GPU buffer to staging
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
  device.queue.submit([encoder.finish()]);

  // Wait for GPU work to complete
  await device.queue.onSubmittedWorkDone();

  // Map and read
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const copyArrayBuffer = stagingBuffer.getMappedRange().slice(0);
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return copyArrayBuffer;
}


export async function readAsFloat32(device, buffer, numElements) {
  const arrayBuffer = await readGPUBuffer(device, buffer, numElements * 4);
  return new Float32Array(arrayBuffer);
}


export async function readAsUint32(device, buffer, numElements) {
  const arrayBuffer = await readGPUBuffer(device, buffer, numElements * 4);
  return new Uint32Array(arrayBuffer);
}


export function uploadToBuffer(device, buffer, data, offset = 0) {
  // Use the ArrayBuffer form to satisfy stricter type checking
  device.queue.writeBuffer(buffer, offset, data.buffer, data.byteOffset, data.byteLength);
}


export function clearBuffer(device, buffer, size) {
  const encoder = device.createCommandEncoder();
  encoder.clearBuffer(buffer, 0, size);
  device.queue.submit([encoder.finish()]);
}
