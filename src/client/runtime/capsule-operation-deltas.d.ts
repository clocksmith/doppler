export type CapsuleOperationDelta = { tokenIds: number[]; text: string } | { itemIndex: number; item: { embedding: number[]; [key: string]: unknown } };
export function createCapsuleDeltaBudget(request: CapsuleOperationRequest): {
  accept(value: unknown): CapsuleOperationDelta;
};
import type { CapsuleOperationRequest } from '../../config/capsule-operation.js';
