import { DOPPLER_VERSION } from '../../version.js';
import { freezeCapsuleV2 } from '../../config/capsule-v2.js';
import { CAPSULE_OPERATION_EVENT_SCHEMA, CAPSULE_OPERATION_RECEIPT_SCHEMA, hashCapsuleObservation, normalizeCapsuleObservation, snapshotCapsuleOperationRequest } from '../../config/capsule-operation.js';

export function createCapsuleOperationExecutor({ adapters, identity, assertCurrent }) {
  const executionIdentity = freezeCapsuleV2(normalizeCapsuleObservation(identity));
  let active = false;
  return (input, { signal: externalSignal = null } = {}) => {
    const request = snapshotCapsuleOperationRequest(input);
    const adapter = adapters[request.operation.name];
    if (!adapter) throw new Error('Selected Capsule runtime has no adapter for this operation.');
    adapter.validate(request);
    const requestHash = hashCapsuleObservation(request);
    const assignmentHash = request.assignment === null ? null : hashCapsuleObservation(request.assignment);
    return (async function* () {
      if (active) throw new Error('Capsule operation already active; submit a distinct job after it finishes.');
      active = true;
      const controller = new AbortController();
      const cancel = () => controller.abort(externalSignal.reason ?? new Error('Capsule operation cancelled.'));
      const deadline = () => controller.abort(new Error('Capsule operation deadline exceeded.'));
      let timer;
      let iterator;
      let failure;
      const check = async () => {
        if (Date.now() >= request.limits.deadlineAt) deadline();
        controller.signal.throwIfAborted();
        await assertCurrent(request);
        if (Date.now() >= request.limits.deadlineAt) deadline();
        controller.signal.throwIfAborted();
      };
      const boundedOutput = (value) => {
        const output = normalizeCapsuleObservation(value);
        if (new TextEncoder().encode(JSON.stringify(output)).length > request.limits.maxOutputBytes) throw new Error('Capsule operation output exceeds maxOutputBytes.');
        return output;
      };
      let eventIndex = 0;
      let previousEventDigest = null;
      const event = (payload) => {
        const value = { schema: CAPSULE_OPERATION_EVENT_SCHEMA, operation: request.operation, requestHash, assignmentHash, eventIndex: eventIndex++, previousEventDigest, ...payload };
        previousEventDigest = hashCapsuleObservation(value);
        return freezeCapsuleV2({ ...value, eventDigest: previousEventDigest });
      };
      try {
        externalSignal?.addEventListener('abort', cancel, { once: true });
        if (externalSignal?.aborted) cancel();
        await check();
        const remaining = request.limits.deadlineAt - Date.now();
        // Timer range is a host API limit, not an execution-policy fallback.
        if (remaining > 2147483647) throw new Error('Capsule operation deadline exceeds the host timer range.');
        timer = setTimeout(deadline, Math.max(0, remaining));
        iterator = adapter.execute(request, controller.signal);
        while (true) {
          await check();
          const step = await iterator.next();
          await check();
          const output = boundedOutput(step.done ? step.value : step.value.output);
          if (step.done) {
            const finishedIterator = iterator;
            iterator = null;
            await finishedIterator.return?.();
            await check();
            const payload = { schema: CAPSULE_OPERATION_RECEIPT_SCHEMA, ...executionIdentity, runtimeVersion: DOPPLER_VERSION,
              operation: request.operation, requestHash, assignmentHash,
              inputHash: hashCapsuleObservation({ input: request.input, options: request.options }), outputHash: hashCapsuleObservation(output) };
            const receipt = freezeCapsuleV2({ ...payload, receiptDigest: hashCapsuleObservation(payload) });
            yield event({ status: 'completed', output, receipt });
            return;
          }
          yield event({ status: 'partial', delta: normalizeCapsuleObservation(step.value.delta), output });
        }
      } catch (error) {
        failure = error;
        throw error;
      } finally {
        controller.abort(new Error('Capsule operation iterator closed.'));
        clearTimeout(timer);
        externalSignal?.removeEventListener('abort', cancel);
        try { await iterator?.return(); } catch (cleanupError) {
          if (failure) throw new AggregateError([failure, cleanupError], failure.message, { cause: failure });
          throw cleanupError;
        } finally { active = false; }
      }
    })();
  };
}
