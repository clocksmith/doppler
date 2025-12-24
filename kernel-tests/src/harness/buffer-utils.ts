/**
 * GPU Buffer Utilities
 */

export type BufferUsage = 'read' | 'write' | 'readwrite';

/**
 * Create a GPU buffer from typed array data
 * @param device GPU device
 * @param data Input data
 * @param usage Buffer usage mode
 * @returns Created GPU buffer
 */
export function createBuffer(
  device: GPUDevice,
  data: Float32Array | Uint32Array | Int32Array,
  usage: BufferUsage = 'read'
): GPUBuffer {
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
  const arrayType = data.constructor as
    | Float32ArrayConstructor
    | Uint32ArrayConstructor
    | Int32ArrayConstructor;
  new arrayType(buffer.getMappedRange()).set(data);
  buffer.unmap();

  return buffer;
}

/**
 * Create an empty GPU buffer
 * @param device GPU device
 * @param size Size in bytes
 * @param usage Buffer usage mode
 * @returns Created GPU buffer
 */
export function createEmptyBuffer(
  device: GPUDevice,
  size: number,
  usage: BufferUsage = 'readwrite'
): GPUBuffer {
  let gpuUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  return device.createBuffer({
    size,
    usage: gpuUsage,
  });
}

/**
 * Read data back from GPU buffer
 * @param device GPU device
 * @param buffer GPU buffer to read from
 * @param size Size in bytes to read
 * @returns Promise resolving to array buffer
 */
export async function readGPUBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  size: number
): Promise<ArrayBuffer> {
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

/**
 * Read GPU buffer as Float32Array
 * @param device GPU device
 * @param buffer GPU buffer to read from
 * @param numElements Number of elements to read
 * @returns Promise resolving to Float32Array
 */
export async function readAsFloat32(
  device: GPUDevice,
  buffer: GPUBuffer,
  numElements: number
): Promise<Float32Array> {
  const arrayBuffer = await readGPUBuffer(device, buffer, numElements * 4);
  return new Float32Array(arrayBuffer);
}

/**
 * Read GPU buffer as Uint32Array
 * @param device GPU device
 * @param buffer GPU buffer to read from
 * @param numElements Number of elements to read
 * @returns Promise resolving to Uint32Array
 */
export async function readAsUint32(
  device: GPUDevice,
  buffer: GPUBuffer,
  numElements: number
): Promise<Uint32Array> {
  const arrayBuffer = await readGPUBuffer(device, buffer, numElements * 4);
  return new Uint32Array(arrayBuffer);
}

/**
 * Upload data to existing GPU buffer
 * @param device GPU device
 * @param buffer Target GPU buffer
 * @param data Data to upload
 * @param offset Byte offset in buffer
 */
export function uploadToBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: Float32Array | Uint32Array,
  offset: number = 0
): void {
  // Use the ArrayBuffer form to satisfy stricter type checking
  device.queue.writeBuffer(buffer, offset, data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Clear a GPU buffer to zeros
 * @param device GPU device
 * @param buffer GPU buffer to clear
 * @param size Size in bytes
 */
export function clearBuffer(device: GPUDevice, buffer: GPUBuffer, size: number): void {
  const encoder = device.createCommandEncoder();
  encoder.clearBuffer(buffer, 0, size);
  device.queue.submit([encoder.finish()]);
}
