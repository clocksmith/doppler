import { createImmediateResourceScope } from '../../../resource-scope.js';

/** @type {import('./decode-components.js').createDecodeStepResources} */
export function createDecodeStepResources(recorder, release, discard) {
  let cleanupUncertain = false;
  /** @type {(buffer: GPUBuffer) => void} */
  const dispose = buffer => cleanupUncertain ? discard(buffer) : release(buffer);
  const resources = createImmediateResourceScope({ release: dispose });
  return { resources, async close(failed, failure) {
    const cleanupErrors = [];
    if (failed) {
      // Abort unsubmitted commands, or await their existing submission owner.
      // Deferred submission owns temporaries until completion even if mapping fails.
      try { await recorder?.abort(); } catch (error) { cleanupErrors.push(error); }
      try { await recorder?.completeDeferredCleanup({ discardPooled: true }); } catch (error) { cleanupErrors.push(error); }
    }
    cleanupUncertain = cleanupErrors.length > 0;
    try { resources.close(failed ? 'failure' : 'success'); } catch (error) { cleanupErrors.push(error); }
    if (cleanupErrors.length) {
      throw new AggregateError(failed ? [failure, ...cleanupErrors] : cleanupErrors,
        'Decode step cleanup failed.', failed ? { cause: failure } : undefined);
    }
  } };
}
