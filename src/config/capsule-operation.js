import catalog from './capsule-operations.json' with { type: 'json' };
import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { freezeCapsuleV2 } from './capsule-v2.js';
import { GENERATION_CONTRACT } from './generation-contract.js';

export const CAPSULE_OPERATION_REQUEST_SCHEMA = 'doppler.capsule-operation-request/v1';
export const CAPSULE_OPERATION_RECEIPT_SCHEMA = 'doppler.capsule-operation-receipt/v1';
export const CAPSULE_OPERATION_EVENT_SCHEMA = 'doppler.capsule-operation-event/v1';
export const CAPSULE_OPERATIONS = freezeCapsuleV2({ ...catalog.operations, generate: {
  ...catalog.operations.generate,
  inputFields: Object.keys(GENERATION_CONTRACT.input),
  optionFields: Object.keys(GENERATION_CONTRACT.options),
} });

// Receipt serialization only: no tensor arithmetic or model-policy selection.
export function normalizeCapsuleObservation(value, depth = 0, ancestors = new Set()) {
  if (depth > catalog.maxObservationDepth) throw new Error('Capsule observation exceeds its depth limit.');
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (!value || typeof value !== 'object' || ancestors.has(value)) throw new Error('Capsule observation must be finite, acyclic JSON data.');
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) value = Array.from(value);
  if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
    throw new Error('Capsule observation contains an unsupported object.');
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) return Array.from(value, (item) => normalizeCapsuleObservation(item, depth + 1, ancestors));
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, normalizeCapsuleObservation(item, depth + 1, ancestors)]));
  } finally { ancestors.delete(value); }
}

export function hashCapsuleObservation(value) {
  return computeCanonicalSha256(normalizeCapsuleObservation(value));
}

export function assertCapsuleOperationFields(value, fields, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
    || Object.keys(value).some((key) => !fields.includes(key))) throw new Error(`Invalid ${label} fields.`);
}

export function snapshotCapsuleOperationRequest(value) {
  const request = normalizeCapsuleObservation(value);
  assertCapsuleOperationFields(request, ['schema', 'operation', 'input', 'options', 'assignment', 'limits', 'adapterSet'], 'Capsule operation request');
  if (request.schema !== CAPSULE_OPERATION_REQUEST_SCHEMA) throw new Error('Unsupported Capsule operation request schema.');
  assertCapsuleOperationFields(request.operation, ['name', 'version'], 'Capsule operation');
  const definition = CAPSULE_OPERATIONS[request.operation.name];
  if (!Object.hasOwn(CAPSULE_OPERATIONS, request.operation.name) || definition.version !== request.operation.version) {
    throw new Error('Unsupported Capsule operation or version.');
  }
  assertCapsuleOperationFields(request.input, definition.inputFields, 'Capsule operation input');
  assertCapsuleOperationFields(request.options, definition.optionFields, 'Capsule operation options');
  if (request.assignment !== null && (!request.assignment || typeof request.assignment !== 'object' || Array.isArray(request.assignment))) {
    throw new Error('Capsule operation assignment must be an object or explicit null.');
  }
  assertCapsuleOperationFields(request.limits, ['maxInputBytes', 'maxOutputBytes', 'deadlineAt'], 'Capsule operation limits');
  for (const key of ['maxInputBytes', 'maxOutputBytes', 'deadlineAt']) {
    if (!Number.isSafeInteger(request.limits[key]) || request.limits[key] <= 0) throw new Error(`Capsule operation requires positive ${key}.`);
  }
  if (new TextEncoder().encode(JSON.stringify({ input: request.input, options: request.options })).length > request.limits.maxInputBytes) {
    throw new Error('Capsule operation input exceeds maxInputBytes.');
  }
  return freezeCapsuleV2(request);
}
