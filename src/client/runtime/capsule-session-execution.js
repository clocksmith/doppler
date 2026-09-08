// One loaded program owns mutable generation and adapter state. Cancellation
// requests cooperation; disposal waits until the current operation has drained.
export function createCapsuleSessionExecution() {
  let active = null;
  let closing = false;
  let closePromise = null;

  function acquire(controller, signal) {
    if (closing) throw new Error('Capsule runtime session is closed.');
    if (active) throw new Error('Capsule operation already active; submit a distinct job after it finishes.');
    if (signal != null && (typeof signal.throwIfAborted !== 'function'
      || typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function'
      || typeof signal.aborted !== 'boolean')) throw new Error('Capsule operation signal must be an AbortSignal.');
    let finish;
    const done = new Promise(resolve => { finish = resolve; });
    const cancel = () => controller.abort(signal.reason);
    const lease = { controller, done, stop: null, release() {
      signal?.removeEventListener('abort', cancel);
      active = null;
      finish();
    } };
    active = lease;
    signal?.addEventListener('abort', cancel, { once: true });
    if (signal?.aborted) cancel();
    return lease;
  }

  return {
    run(task, signal) {
      const controller = new AbortController();
      const lease = acquire(controller, signal);
      try {
        controller.signal.throwIfAborted();
        const result = task(controller.signal);
        if (result && typeof result.then === 'function') {
          return Promise.resolve(result).then(value => {
            controller.signal.throwIfAborted();
            return value;
          }).finally(lease.release);
        }
        controller.signal.throwIfAborted();
        lease.release();
        return result;
      } catch (error) {
        lease.release();
        throw error;
      }
    },

    stream(create, signal) {
      const controller = new AbortController();
      // Snapshot/validate requests synchronously, before a caller can edit them.
      const source = create(controller.signal);
      const output = (async function* () {
        const lease = acquire(controller, signal);
        lease.stop = () => output.return();
        try {
          controller.signal.throwIfAborted();
          while (true) {
            const step = await source.next();
            controller.signal.throwIfAborted();
            if (step.done) return step.value;
            yield step.value;
          }
        } finally {
          try { await source.return?.(); } finally { lease.release(); }
        }
      })();
      return output;
    },

    close(dispose) {
      if (closePromise) return closePromise;
      closing = true;
      const current = active;
      current?.controller.abort(new Error('Capsule runtime session is closed.'));
      closePromise = (async () => {
        const errors = [];
        try { await current?.stop?.(); } catch (error) { errors.push(error); }
        await current?.done;
        try { await dispose(); } catch (error) { errors.push(error); }
        if (errors.length === 1) throw errors[0];
        if (errors.length) throw new AggregateError(errors, errors[0].message, { cause: errors[0] });
      })();
      return closePromise;
    },
  };
}
