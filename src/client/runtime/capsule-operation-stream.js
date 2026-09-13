import { hashCapsuleObservation, normalizeCapsuleObservation, resolveCapsuleStreamFormat, snapshotCapsuleOperationRequest } from '../../config/capsule-operation.js';
import { freezeCapsuleV2 } from '../../config/capsule-v2.js';
import { createCapsuleDeltaBudget } from './capsule-operation-deltas.js';

const requireValue = (value, message) => { if (!value) throw new Error(`Capsule stream: ${message}`); };

// Transport integrity and reconstruction, not application trust or qualification.
// accept() never materializes a cumulative snapshot for a partial event.
export function createCapsuleStreamAccumulator(requestInput) {
  const request = snapshotCapsuleOperationRequest(requestInput);
  const format = resolveCapsuleStreamFormat(request.schema);
  requireValue(format.incremental, 'incremental reconstruction requires operation request v2');
  const requestHash = hashCapsuleObservation(request);
  const assignmentHash = request.assignment === null ? null : hashCapsuleObservation(request.assignment);
  const inputHash = hashCapsuleObservation({ input: request.input, options: request.options });
  const budget = createCapsuleDeltaBudget(request);
  const tokenIds = [], textParts = [], embeddings = [];
  let eventIndex = 0, previousEventDigest = null, completed = null;
  let failure = null;
  const partialOutput = () => request.operation.name === 'generate'
    ? { text: textParts.join(''), tokenIds: [...tokenIds] } : { embeddings: [...embeddings] };
  return {
    accept(value) {
      if (failure) throw failure;
      try {
        requireValue(!completed, 'event after completion');
        const event = normalizeCapsuleObservation(value);
        const { eventDigest, ...payload } = event;
        requireValue(event.schema === format.eventSchema && ['partial', 'completed'].includes(event.status), 'event format mismatch');
        requireValue(event.eventIndex === eventIndex && event.previousEventDigest === previousEventDigest, 'duplicate, missing, or reordered event');
        requireValue(event.requestHash === requestHash && event.assignmentHash === assignmentHash
          && event.operation?.name === request.operation.name && event.operation?.version === request.operation.version, 'event request mismatch');
        requireValue(eventDigest === hashCapsuleObservation(payload), 'event digest mismatch');
        if (event.status === 'partial') {
          requireValue(!Object.hasOwn(event, 'output') && !Object.hasOwn(event, 'receipt'), 'partial event must contain only its delta');
          const delta = budget.accept(event.delta);
          if (request.operation.name === 'generate') {
            for (const id of delta.tokenIds) tokenIds.push(id);
            if (delta.text) textParts.push(delta.text);
          } else embeddings.push(delta.item);
        } else {
          requireValue(!Object.hasOwn(event, 'delta'), 'completion cannot contain a delta');
          const { receiptDigest, ...receipt } = event.receipt ?? {};
          requireValue(receipt.schema === format.receiptSchema && receiptDigest === hashCapsuleObservation(receipt), 'completion receipt digest mismatch');
          requireValue(receipt.stream?.schema === format.eventSchema && receipt.stream.partialCount === eventIndex
            && receipt.stream.lastPartialDigest === previousEventDigest, 'completion does not bind the partial stream');
          requireValue(receipt.requestHash === requestHash && receipt.assignmentHash === assignmentHash && receipt.inputHash === inputHash
            && receipt.operation?.name === request.operation.name && receipt.operation?.version === request.operation.version, 'receipt request mismatch');
          requireValue(new TextEncoder().encode(JSON.stringify(event.output)).length <= request.limits.maxOutputBytes, 'output exceeds maxOutputBytes');
          requireValue(receipt.outputHash === hashCapsuleObservation(event.output), 'completion output digest mismatch');
          if (request.operation.name === 'generate' || request.operation.name === 'embed') {
            const reconstructed = { ...event.output, ...partialOutput() };
            requireValue(hashCapsuleObservation(reconstructed) === receipt.outputHash, 'completion differs from reconstructed output');
          }
          completed = freezeCapsuleV2(event);
        }
        eventIndex++;
        previousEventDigest = eventDigest;
      } catch (error) { failure = error; throw error; }
    },
    // Explicit compatibility cost: callers choose when to copy a full snapshot.
    snapshot() {
      if (failure) throw failure;
      return completed?.output ?? freezeCapsuleV2(partialOutput());
    },
    finish() {
      if (failure) throw failure;
      requireValue(completed, 'stream ended without completion');
      return completed;
    },
  };
}

export async function* capsuleOperationSnapshots(events, request) {
  const accumulator = createCapsuleStreamAccumulator(request);
  for await (const event of events) {
    accumulator.accept(event);
    yield { event, output: accumulator.snapshot() };
  }
  accumulator.finish();
}
