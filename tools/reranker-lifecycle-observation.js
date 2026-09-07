// Browser-only observation helpers. They never alter submitted commands or math.
export function captureRequestedDevices() {
  const original = GPUAdapter.prototype.requestDevice;
  const devices = [];
  GPUAdapter.prototype.requestDevice = async function (...args) {
    const device = await original.apply(this, args);
    devices.push(device);
    return device;
  };
  return { devices, restore() { GPUAdapter.prototype.requestDevice = original; } };
}

export async function observeSettled(operation, timeoutMs) {
  let timer;
  const started = performance.now();
  try {
    const value = await Promise.race([
      Promise.resolve().then(operation),
      new Promise((resolve, reject) => {
        timer = setTimeout(() => reject(new Error('Observation deadline exceeded.')), timeoutMs);
      }),
    ]);
    return { status: 'fulfilled', elapsedMs: performance.now() - started, value };
  } catch (error) {
    return { status: error.message === 'Observation deadline exceeded.' ? 'timeout' : 'rejected',
      elapsedMs: performance.now() - started, error: { name: error.name, message: error.message } };
  } finally { clearTimeout(timer); }
}

export async function observeSubmittedCancellation(run, devices, timeoutMs) {
  const controller = new AbortController();
  let submissions = 0, cancelledAt = null, submittedDevice = null;
  const restore = [];
  let outcome;
  try {
    for (const device of devices) {
      const queue = device.queue;
      const descriptor = Object.getOwnPropertyDescriptor(queue, 'submit');
      const original = queue.submit;
      queue.submit = function (...args) {
        const result = original.apply(this, args);
        submissions++;
        if (!controller.signal.aborted) {
          submittedDevice = device;
          cancelledAt = performance.now();
          controller.abort(new DOMException('Qualification cancellation after GPU submission.', 'AbortError'));
        }
        return result;
      };
      restore.push(() => {
        if (descriptor) Object.defineProperty(queue, 'submit', descriptor);
        else delete queue.submit;
      });
    }
    outcome = await observeSettled(() => run(controller.signal), timeoutMs);
  } finally { for (const reset of restore.reverse()) reset(); }
  return { device: submittedDevice, observation: { submissions, triggeredAfterSubmission: cancelledAt !== null,
    completedAfterAbortMs: cancelledAt === null ? null : performance.now() - cancelledAt,
    cancellationHonored: outcome.status === 'rejected' && outcome.error.name === 'AbortError', outcome } };
}
