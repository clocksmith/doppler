import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const REFACTOR_RECEIPT_SCHEMA = 'doppler.refactor-receipt/v1';

function assertJsonSafe(value, path = 'receipt') {
  if (value === null) {
    return;
  }
  const valueType = typeof value;
  if (valueType === 'string' || valueType === 'boolean') {
    return;
  }
  if (valueType === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error(`[RefactorReceipt] ${path} must contain only finite numbers.`);
    }
    return;
  }
  if (valueType === 'undefined' || valueType === 'function' || valueType === 'symbol' || valueType === 'bigint') {
    throw new Error(`[RefactorReceipt] ${path} is not JSON-safe.`);
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      assertJsonSafe(value[index], `${path}[${index}]`);
    }
    return;
  }
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    throw new Error(`[RefactorReceipt] ${path} must describe tensor data instead of embedding binary data.`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new Error(`[RefactorReceipt] ${path} must contain plain objects only.`);
  }
  for (const [key, entry] of Object.entries(value)) {
    assertJsonSafe(entry, `${path}.${key}`);
  }
}

function deepFreeze(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) {
    return value;
  }
  Object.freeze(value);
  for (const entry of Object.values(value)) {
    deepFreeze(entry);
  }
  return value;
}

export function canonicalizeRefactorReceiptValue(value) {
  assertJsonSafe(value);
  return stableSortObject(value);
}

export function hashRefactorReceiptValue(value) {
  const canonical = canonicalizeRefactorReceiptValue(value);
  return `sha256:${sha256Hex(JSON.stringify(canonical))}`;
}

function normalizeOptionalRecord(value, label) {
  if (value === undefined || value === null) {
    return null;
  }
  assertJsonSafe(value, label);
  return stableSortObject(value);
}

function normalizeEventList(value, label) {
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new Error(`[RefactorReceipt] ${label} must be an array.`);
  }
  assertJsonSafe(value, label);
  return value.map((entry) => stableSortObject(entry));
}

export function createRefactorReceipt(input = {}) {
  const commandContext = normalizeOptionalRecord(input.commandContext, 'commandContext');
  const resolvedSession = normalizeOptionalRecord(input.resolvedSession, 'resolvedSession');
  const observationContext = normalizeOptionalRecord(input.observationContext, 'observationContext');
  const operationPlan = normalizeOptionalRecord(input.operationPlan, 'operationPlan');
  const operations = normalizeEventList(input.operations, 'operations');
  const dtypeTransitions = normalizeEventList(input.dtypeTransitions, 'dtypeTransitions');
  const resourceEvents = normalizeEventList(input.resourceEvents, 'resourceEvents');
  const failure = normalizeOptionalRecord(input.failure, 'failure');

  const semanticCore = {
    resolvedSession,
    operationPlan,
  };
  const executionCore = {
    operations,
    dtypeTransitions,
    resourceEvents,
    failure,
  };
  const core = {
    schema: REFACTOR_RECEIPT_SCHEMA,
    commandContext,
    resolvedSession,
    observationContext,
    operationPlan,
    operations,
    dtypeTransitions,
    resourceEvents,
    failure,
    semanticHash: hashRefactorReceiptValue(semanticCore),
    executionHash: hashRefactorReceiptValue(executionCore),
  };
  const receipt = {
    ...core,
    receiptHash: hashRefactorReceiptValue(core),
  };
  return deepFreeze(receipt);
}

export function verifyRefactorReceipt(receipt) {
  if (!receipt || receipt.schema !== REFACTOR_RECEIPT_SCHEMA) {
    throw new Error(`[RefactorReceipt] schema must be "${REFACTOR_RECEIPT_SCHEMA}".`);
  }
  const {
    semanticHash,
    executionHash,
    receiptHash,
    ...core
  } = receipt;
  const expectedSemanticHash = hashRefactorReceiptValue({
    resolvedSession: receipt.resolvedSession,
    operationPlan: receipt.operationPlan,
  });
  const expectedExecutionHash = hashRefactorReceiptValue({
    operations: receipt.operations,
    dtypeTransitions: receipt.dtypeTransitions,
    resourceEvents: receipt.resourceEvents,
    failure: receipt.failure,
  });
  const expectedReceiptHash = hashRefactorReceiptValue({
    ...core,
    semanticHash,
    executionHash,
  });
  if (semanticHash !== expectedSemanticHash) {
    throw new Error('[RefactorReceipt] semanticHash mismatch.');
  }
  if (executionHash !== expectedExecutionHash) {
    throw new Error('[RefactorReceipt] executionHash mismatch.');
  }
  if (receiptHash !== expectedReceiptHash) {
    throw new Error('[RefactorReceipt] receiptHash mismatch.');
  }
  return true;
}

export function compareRefactorReceipts(expected, actual) {
  verifyRefactorReceipt(expected);
  verifyRefactorReceipt(actual);
  const fields = [
    'semanticHash',
    'executionHash',
    'commandContext',
    'resolvedSession',
    'observationContext',
    'operationPlan',
    'operations',
    'dtypeTransitions',
    'resourceEvents',
    'failure',
  ];
  const differences = [];
  for (const field of fields) {
    if (
      JSON.stringify(canonicalizeRefactorReceiptValue(expected[field]))
      !== JSON.stringify(canonicalizeRefactorReceiptValue(actual[field]))
    ) {
      differences.push(field);
    }
  }
  return deepFreeze({
    matches: differences.length === 0,
    semanticMatches: expected.semanticHash === actual.semanticHash,
    executionMatches: expected.executionHash === actual.executionHash,
    differences,
  });
}
