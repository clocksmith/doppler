import type { PackOperationRequest } from '../../config/pack-operation.js';
import type { PackOperationAdapter } from './pack-operation-adapters.js';
export interface PackOperationEvent {
  schema: 'doppler.pack-operation-event/v1';
  operation: PackOperationRequest['operation'];
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
export function createPackOperationExecutor(ports: {
  adapters: Record<string, PackOperationAdapter>;
  identity: Record<string, unknown>;
  assertCurrent(): Promise<void>;
}): (request: PackOperationRequest, control?: { signal?: AbortSignal | null }) => AsyncGenerator<PackOperationEvent, void, void>;
