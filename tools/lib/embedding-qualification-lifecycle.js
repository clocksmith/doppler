// Browser-safe qualification control; the runtime continues to own execution/cleanup.
export async function qualifyEmbeddingCancellation({ queue, run, resources }) {
  const before = structuredClone(resources());
  const controller = new AbortController();
  const reason = new Error('Embedding qualification cancellation after GPU submission');
  const originalSubmit = queue.submit;
  let submissions = 0;
  let rejection = null;
  queue.submit = function (...args) {
    const result = originalSubmit.apply(this, args);
    submissions++;
    if (!controller.signal.aborted) controller.abort(reason);
    return result;
  };
  try { await run(controller.signal); }
  catch (error) { rejection = error; }
  finally {
    queue.submit = originalSubmit;
    await queue.onSubmittedWorkDone();
  }
  const after = structuredClone(resources());
  if (submissions < 1 || !controller.signal.aborted) throw new Error('Embedding cancellation did not follow a GPU submission.');
  if (!rejection || (rejection !== reason && rejection.name !== 'AbortError' && !String(rejection.message).includes(reason.message))) {
    throw new Error(`Embedding cancellation did not propagate its abort: ${rejection?.message ?? 'completed output'}`);
  }
  if (after.deferredCleanup.bytes !== 0 || after.deferredCleanup.count !== 0
    || after.active.bytes > before.active.bytes || after.active.count > before.active.count) {
    throw new Error('Cancelled embedding retained active or deferred GPU resources.');
  }
  return { passed: true, submissions, rejected: { name: rejection.name, message: rejection.message }, before, after };
}
