// Browser-side diagnostic observation. No extra submission, fence or readback.
export function installGpuObservation() {
  const restores = [];
  let active = false;
  let startedAt = 0;
  let rows = [];
  let counts = {};
  const now = () => performance.now();
  const hook = (owner, name, observe) => {
    const descriptor = Object.getOwnPropertyDescriptor(owner, name);
    const original = owner[name];
    Object.defineProperty(owner, name, { ...descriptor, value: function (...args) {
      if (!active) return original.apply(this, args);
      return observe.call(this, original, args);
    } });
    restores.push(() => Object.defineProperty(owner, name, descriptor));
  };
  const countCall = (owner, name, counter) => hook(owner, name, function (original, args) {
    const start = now();
    const result = original.apply(this, args);
    counts[counter] = (counts[counter] ?? 0) + 1;
    counts[`${counter}CpuMs`] = (counts[`${counter}CpuMs`] ?? 0) + now() - start;
    return result;
  });
  const observeWait = (owner, name, kind) => hook(owner, name, function (original, args) {
    const start = now();
    const currentRows = rows;
    const origin = startedAt;
    const label = this.label;
    const result = original.apply(this, args);
    const finish = failed => currentRows.push({ kind, label, startMs: start - origin,
      durationMs: now() - start, failed });
    result.then(() => finish(false), () => finish(true));
    return result;
  });
  try {
    countCall(GPUQueue.prototype, 'submit', 'submissions');
    countCall(GPUComputePassEncoder.prototype, 'dispatchWorkgroups', 'dispatches');
    countCall(GPUComputePassEncoder.prototype, 'dispatchWorkgroupsIndirect', 'indirectDispatches');
    observeWait(GPUQueue.prototype, 'onSubmittedWorkDone', 'queue-wait');
    observeWait(GPUBuffer.prototype, 'mapAsync', 'map-wait');
    hook(GPUCommandEncoder.prototype, 'copyBufferToBuffer', function (original, args) {
      const result = original.apply(this, args);
      const [source, , destination, , bytes] = args;
      if ((destination.usage & GPUBufferUsage.MAP_READ) !== 0) {
        rows.push({ kind: 'readback-copy', startMs: now() - startedAt, bytes,
          source: source.label, destination: destination.label });
      }
      return result;
    });
  } catch (error) {
    for (const restore of restores.reverse()) restore();
    throw error;
  }
  return {
    start() { rows = []; counts = {}; startedAt = now(); active = true; },
    snapshot() { return { elapsedMs: now() - startedAt, counts: { ...counts }, rows: rows.length }; },
    stop() { active = false; return { counts: { ...counts }, rows: [...rows] }; },
    restore() { active = false; for (const restore of restores.reverse()) restore(); },
  };
}
