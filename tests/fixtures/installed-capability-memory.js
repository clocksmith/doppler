// Browser-side observation only. No allocation sizes, results, or ownership rules
// are changed. Counts describe live WebGPU buffer objects, not driver residency.
export function installCapabilityMemoryProbe() {
  const metrics = { createdBytes: 0, destroyedBytes: 0, liveBytes: 0, peakLiveBytes: 0,
    createdCount: 0, liveCount: 0, failedAllocations: 0 };
  const buffers = new WeakMap();
  const devices = new WeakMap();
  const liveLabels = new Map();
  function changeLabel(labels, label, bytes, count) {
    const entry = labels.get(label) ?? { bytes: 0, count: 0 };
    entry.bytes += bytes; entry.count += count;
    if (entry.count) labels.set(label, entry); else labels.delete(label);
  }
  function closeDevice(state) {
    if (!state || state.closed) return;
    metrics.destroyedBytes += state.bytes;
    metrics.liveBytes -= state.bytes;
    metrics.liveCount -= state.count;
    for (const [label, entry] of state.labels) changeLabel(liveLabels, label, -entry.bytes, -entry.count);
    state.labels.clear();
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
      device = { bytes: 0, count: 0, closed: false, labels: new Map() };
      devices.set(this, device);
      this.lost.then(() => closeDevice(device));
    }
    const size = buffer.size;
    const label = descriptor.label ?? '';
    buffers.set(buffer, { size, label, device, destroyed: false });
    changeLabel(liveLabels, label, size, 1);
    changeLabel(device.labels, label, size, 1);
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
      changeLabel(liveLabels, buffer.label, -buffer.size, -1);
      changeLabel(buffer.device.labels, buffer.label, -buffer.size, -1);
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
    // Allocation sites, not an inference that a label establishes current ownership.
    liveBufferLabels: [...liveLabels].map(([label, entry]) => ({ label, ...entry })),
    jsHeapUsedBytes: performance.memory?.usedJSHeapSize ?? null,
    scope: 'Observed WebGPU object lifetimes and browser-reported JS heap; not physical driver residency or hashing workspace.',
  });
}
