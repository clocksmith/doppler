import type { CapsuleAdapterArtifactStore, PreparedCapsuleAdapterExecution } from './capsule-adapter-execution.js';
import type { CapsuleOperationRequest } from '../../config/capsule-operation.js';
import type { CapsuleOperationAdapter } from './capsule-operation-adapters.js';
export interface CapsuleOperationEvent {
  schema: 'doppler.capsule-operation-event/v1';
  operation: CapsuleOperationRequest['operation'];
  requestHash: string;
  assignmentHash: string | null;
  eventIndex: number;
  previousEventDigest: string | null;
  eventDigest: string;
  status: 'partial' | 'completed';
  delta?: unknown;
  output: unknown;
  receipt?: Record<string, unknown>;
}
export function createCapsuleOperationExecutor(ports: {
  adapters: Record<string, CapsuleOperationAdapter>;
  identity: Record<string, unknown>;
  assertCurrent(request: CapsuleOperationRequest): Promise<void>;
  prepareExecution?: ((request: CapsuleOperationRequest, control: {
    signal: AbortSignal;
    adapterArtifactStore: CapsuleAdapterArtifactStore | null;
  }) => Promise<PreparedCapsuleAdapterExecution | null>) | null;
}): (request: CapsuleOperationRequest, control?: { signal?: AbortSignal | null; adapterArtifactStore?: CapsuleAdapterArtifactStore | null }) => AsyncGenerator<CapsuleOperationEvent, void, void>;
