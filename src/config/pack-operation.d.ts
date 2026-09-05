export const PACK_OPERATION_REQUEST_SCHEMA: 'doppler.pack-operation-request/v1';
export const PACK_OPERATION_RECEIPT_SCHEMA: 'doppler.pack-operation-receipt/v1';
export const PACK_OPERATION_EVENT_SCHEMA: 'doppler.pack-operation-event/v1';
export type PackOperationName = 'generate' | 'embed' | 'rerank' | 'encodeSequence';
export interface PackOperationRequest {
  schema: typeof PACK_OPERATION_REQUEST_SCHEMA;
  operation: { name: PackOperationName; version: 1 };
  input: Record<string, unknown>;
  options: Record<string, unknown>;
  assignment: Record<string, unknown> | null;
  limits: { maxInputBytes: number; maxOutputBytes: number; deadlineAt: number };
}
export const PACK_OPERATIONS: Readonly<Record<PackOperationName, { version: number; inputFields: string[]; optionFields: string[] }>>;
export function normalizePackObservation(value: unknown, depth?: number, ancestors?: Set<object>): unknown;
export function hashPackObservation(value: unknown): string;
export function assertPackOperationFields(value: unknown, fields: string[], label: string): void;
export function snapshotPackOperationRequest(value: unknown): Readonly<PackOperationRequest>;
