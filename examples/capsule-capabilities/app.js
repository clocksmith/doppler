import * as host from 'doppler-gpu/host';

// A versioned descriptor carries the reviewed model, trust and application binding.
// The same public session executes each model's explicitly qualified capability.
export async function runCapability(descriptor, { signal, onProgress, onEvent, persistReleaseCheckpoint } = {}) {
  if (descriptor.schema !== 'doppler.capability-example/v1') {
    throw new Error('Expected a doppler.capability-example/v1 model descriptor.');
  }
  const incremental = descriptor.request.schema === 'doppler.capsule-operation-request/v2';
  if (incremental && typeof host.createCapsuleStreamAccumulator !== 'function') {
    throw new Error('This installed Doppler archive does not support operation request v2.');
  }
  const session = await host.openCapsule(descriptor.capsuleUrl, {
    ...descriptor.openOptions, signal, persistReleaseCheckpoint,
    observer: onProgress ? { observe: onProgress } : undefined,
  });
  try {
    const request = { ...descriptor.request, limits: { ...descriptor.request.limits,
      deadlineAt: Date.now() + descriptor.maxDurationMs } };
    const stream = incremental ? host.createCapsuleStreamAccumulator(request) : null;
    let completed;
    for await (const event of session.executeOperation(request, { signal })) {
      stream?.accept(event);
      await onEvent?.(event);
      if (event.status === 'completed') completed = event;
    }
    if (!completed) throw new Error('The operation ended without a completion record.');
    return stream?.finish() ?? completed;
  } finally {
    await session.close();
  }
}
