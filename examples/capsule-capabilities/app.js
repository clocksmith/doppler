import { openCapsule } from 'doppler-gpu/host';

// A versioned descriptor carries the reviewed model, trust and application binding.
// The same public session executes each model's explicitly qualified capability.
export async function runCapability(descriptor, { signal, onProgress, onEvent, persistReleaseCheckpoint } = {}) {
  if (descriptor.schema !== 'doppler.capability-example/v1') {
    throw new Error('Expected a doppler.capability-example/v1 model descriptor.');
  }
  const session = await openCapsule(descriptor.capsuleUrl, {
    ...descriptor.openOptions, signal, persistReleaseCheckpoint,
    observer: onProgress ? { observe: onProgress } : undefined,
  });
  try {
    let completed;
    const request = { ...descriptor.request, limits: { ...descriptor.request.limits,
      deadlineAt: Date.now() + descriptor.maxDurationMs } };
    for await (const event of session.executeOperation(request, { signal })) {
      await onEvent?.(event);
      if (event.status === 'completed') completed = event;
    }
    if (!completed) throw new Error('The operation ended without a completion record.');
    return completed;
  } finally {
    await session.close();
  }
}
