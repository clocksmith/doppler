// Installed in the qualification page before application code. Observes native
// upload calls without changing their arguments, return values, or ordering.
export function observeDocumentSearchGpu() {
  const metrics = { writeBufferBytes: 0, writeBufferCalls: 0, writeBufferCpuMs: 0, mappedAtCreationBytes: 0 };
  globalThis.documentSearchGpuMetrics = metrics;
  const write = GPUQueue.prototype.writeBuffer;
  GPUQueue.prototype.writeBuffer = function(buffer, offset, data, dataOffset, size) {
    const started = performance.now();
    const result = write.call(this, buffer, offset, data, dataOffset, size);
    metrics.writeBufferCpuMs += performance.now() - started;
    const elementSize = data.BYTES_PER_ELEMENT ?? 1;
    metrics.writeBufferBytes += size === undefined ? data.byteLength - (dataOffset ?? 0) * elementSize : size * elementSize;
    metrics.writeBufferCalls++;
    return result;
  };
  const create = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function(descriptor) {
    const result = create.call(this, descriptor);
    if (descriptor.mappedAtCreation) metrics.mappedAtCreationBytes += descriptor.size;
    return result;
  };
}

export async function readDocumentSearchGpuObservation() {
  const { getDevice } = await import('./runtime/src/gpu/device.js');
  const started = performance.now();
  await getDevice().queue.onSubmittedWorkDone();
  return { ...globalThis.documentSearchGpuMetrics, completionWaitMs: performance.now() - started,
    scope: 'Native upload submissions and queue completion wait; not isolated GPU transfer throughput.' };
}
