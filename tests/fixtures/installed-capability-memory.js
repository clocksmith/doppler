// Browser-side observation only. No allocation sizes, results, or ownership rules
// are changed. Counts describe live WebGPU buffer objects, not driver residency.
export function installCapabilityMemoryProbe() {
  const metrics = { createdBytes: 0, destroyedBytes: 0, liveBytes: 0, peakLiveBytes: 0,
    createdCount: 0, liveCount: 0, failedAllocations: 0 };
  const buffers = new WeakMap();
  const devices = new WeakMap();
  function closeDevice(state) {
    if (!state || state.closed) return;
    metrics.destroyedBytes += state.bytes;
    metrics.liveBytes -= state.bytes;
    metrics.liveCount -= state.count;
    state.closed = true;
  }
  const create = GPUDevice.prototype.createBuffer;
  const destroyBuffer = GPUBuffer.prototype.destroy;
  const destroyDevice = GPUDevice.prototype.destroy;
  GPUDevice.prototype.createBuffer = function(descriptor) {
    let buffer;
    try { buffer = create.call(this, descriptor); }
    catch (error) { metrics.failedAllocations++; throw error; }
    let device = devices.get(this);
    if (!device) {
      device = { bytes: 0, count: 0, closed: false };
      devices.set(this, device);
      this.lost.then(() => closeDevice(device));
    }
    const size = buffer.size;
    buffers.set(buffer, { size, device, destroyed: false });
    device.bytes += size; device.count++;
    metrics.createdBytes += size; metrics.createdCount++;
    metrics.liveBytes += size; metrics.liveCount++;
    metrics.peakLiveBytes = Math.max(metrics.peakLiveBytes, metrics.liveBytes);
    return buffer;
  };
  GPUBuffer.prototype.destroy = function() {
    const result = destroyBuffer.call(this);
    const buffer = buffers.get(this);
    if (buffer && !buffer.destroyed && !buffer.device.closed) {
      buffer.destroyed = true;
      buffer.device.bytes -= buffer.size; buffer.device.count--;
      metrics.destroyedBytes += buffer.size;
      metrics.liveBytes -= buffer.size; metrics.liveCount--;
    }
    return result;
  };
  GPUDevice.prototype.destroy = function() {
    const result = destroyDevice.call(this);
    closeDevice(devices.get(this));
    return result;
  };
  globalThis.readCapabilityMemory = () => ({
    gpuBuffers: { ...metrics },
    jsHeapUsedBytes: performance.memory?.usedJSHeapSize ?? null,
    scope: 'Observed WebGPU object lifetimes and browser-reported JS heap; not physical driver residency or hashing workspace.',
  });
}
