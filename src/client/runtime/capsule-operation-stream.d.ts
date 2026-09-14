import type { CapsuleOperationRequest } from '../../config/capsule-operation.js';
import type { CapsuleOperationEvent } from './capsule-operation-executor.js';
export function createCapsuleStreamAccumulator(request: CapsuleOperationRequest): {
  accept(event: CapsuleOperationEvent): void;
  snapshot(): unknown;
  finish(): CapsuleOperationEvent;
};
export function capsuleOperationSnapshots(events: AsyncIterable<CapsuleOperationEvent>, request: CapsuleOperationRequest): AsyncGenerator<{
  event: CapsuleOperationEvent; output: unknown;
}, void, void>;
