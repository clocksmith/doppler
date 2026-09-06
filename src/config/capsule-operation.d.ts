import type { CapsuleExecutionAdapter } from './capsule-adapters.js';
export const CAPSULE_OPERATION_REQUEST_SCHEMA: 'doppler.capsule-operation-request/v1';
export const CAPSULE_OPERATION_RECEIPT_SCHEMA: 'doppler.capsule-operation-receipt/v1';
export const CAPSULE_OPERATION_EVENT_SCHEMA: 'doppler.capsule-operation-event/v1';
export type CapsuleOperationName = 'generate' | 'embed' | 'rerank' | 'encodeSequence';
export interface CapsuleOperationRequest {
  schema: typeof CAPSULE_OPERATION_REQUEST_SCHEMA;
  operation: { name: CapsuleOperationName; version: 1 };
  input: Record<string, unknown>;
  options: Record<string, unknown>;
  adapterSet?: readonly CapsuleExecutionAdapter[];
  assignment: Record<string, unknown> | null;
  limits: { maxInputBytes: number; maxOutputBytes: number; deadlineAt: number };
}
export const CAPSULE_OPERATIONS: Readonly<Record<CapsuleOperationName, { version: number; inputFields: string[]; optionFields: string[] }>>;
export function normalizeCapsuleObservation(value: unknown, depth?: number, ancestors?: Set<object>): unknown;
export function hashCapsuleObservation(value: unknown): string;
export function assertCapsuleOperationFields(value: unknown, fields: string[], label: string): void;
export function snapshotCapsuleOperationRequest(value: unknown): Readonly<CapsuleOperationRequest>;
