export const REFACTOR_RECEIPT_SCHEMA: 'doppler.refactor-receipt/v1';

export type RefactorReceiptValue =
  | null
  | string
  | number
  | boolean
  | RefactorReceiptValue[]
  | { [key: string]: RefactorReceiptValue };

export interface RefactorReceiptInput {
  commandContext?: Record<string, RefactorReceiptValue> | null;
  resolvedSession?: Record<string, RefactorReceiptValue> | null;
  observationContext?: Record<string, RefactorReceiptValue> | null;
  operationPlan?: Record<string, RefactorReceiptValue> | null;
  operations?: Array<Record<string, RefactorReceiptValue>> | null;
  dtypeTransitions?: Array<Record<string, RefactorReceiptValue>> | null;
  resourceEvents?: Array<Record<string, RefactorReceiptValue>> | null;
  failure?: Record<string, RefactorReceiptValue> | null;
}

export interface RefactorReceipt {
  readonly schema: typeof REFACTOR_RECEIPT_SCHEMA;
  readonly commandContext: Record<string, RefactorReceiptValue> | null;
  readonly resolvedSession: Record<string, RefactorReceiptValue> | null;
  readonly observationContext: Record<string, RefactorReceiptValue> | null;
  readonly operationPlan: Record<string, RefactorReceiptValue> | null;
  readonly operations: ReadonlyArray<Record<string, RefactorReceiptValue>>;
  readonly dtypeTransitions: ReadonlyArray<Record<string, RefactorReceiptValue>>;
  readonly resourceEvents: ReadonlyArray<Record<string, RefactorReceiptValue>>;
  readonly failure: Record<string, RefactorReceiptValue> | null;
  readonly semanticHash: string;
  readonly executionHash: string;
  readonly receiptHash: string;
}

export function canonicalizeRefactorReceiptValue<T extends RefactorReceiptValue>(value: T): T;
export function hashRefactorReceiptValue(value: RefactorReceiptValue): string;
export function createRefactorReceipt(input?: RefactorReceiptInput): RefactorReceipt;
export function verifyRefactorReceipt(receipt: RefactorReceipt): true;
export function compareRefactorReceipts(
  expected: RefactorReceipt,
  actual: RefactorReceipt
): {
  readonly matches: boolean;
  readonly semanticMatches: boolean;
  readonly executionMatches: boolean;
  readonly differences: readonly string[];
};
