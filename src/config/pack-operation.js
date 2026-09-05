import catalog from './pack-operations.json' with { type: 'json' };
import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { freezePackV2 } from './pack-v2.js';

export const PACK_OPERATION_REQUEST_SCHEMA = 'doppler.pack-operation-request/v1';
export const PACK_OPERATION_RECEIPT_SCHEMA = 'doppler.pack-operation-receipt/v1';
export const PACK_OPERATION_EVENT_SCHEMA = 'doppler.pack-operation-event/v1';
export const PACK_OPERATIONS = freezePackV2(catalog.operations);

// Receipt serialization only: no tensor arithmetic or model-policy selection.
export function normalizePackObservation(value, depth = 0, ancestors = new Set()) {
  if (depth > catalog.maxObservationDepth) throw new Error('Pack observation exceeds its depth limit.');
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (!value || typeof value !== 'object' || ancestors.has(value)) throw new Error('Pack observation must be finite, acyclic JSON data.');
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) value = Array.from(value);
  if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
    throw new Error('Pack observation contains an unsupported object.');
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) return Array.from(value, (item) => normalizePackObservation(item, depth + 1, ancestors));
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, normalizePackObservation(item, depth + 1, ancestors)]));
  } finally { ancestors.delete(value); }
}

export function hashPackObservation(value) {
  return computeCanonicalSha256(normalizePackObservation(value));
}

export function assertPackOperationFields(value, fields, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
    || Object.keys(value).some((key) => !fields.includes(key))) throw new Error(`Invalid ${label} fields.`);
}

export function snapshotPackOperationRequest(value) {
  const request = normalizePackObservation(value);
  assertPackOperationFields(request, ['schema', 'operation', 'input', 'options', 'assignment', 'limits'], 'Pack operation request');
  if (request.schema !== PACK_OPERATION_REQUEST_SCHEMA) throw new Error('Unsupported Pack operation request schema.');
  assertPackOperationFields(request.operation, ['name', 'version'], 'Pack operation');
  const definition = PACK_OPERATIONS[request.operation.name];
  if (!Object.hasOwn(PACK_OPERATIONS, request.operation.name) || definition.version !== request.operation.version) {
    throw new Error('Unsupported Pack operation or version.');
  }
  assertPackOperationFields(request.input, definition.inputFields, 'Pack operation input');
  assertPackOperationFields(request.options, definition.optionFields, 'Pack operation options');
  if (request.assignment !== null && (!request.assignment || typeof request.assignment !== 'object' || Array.isArray(request.assignment))) {
    throw new Error('Pack operation assignment must be an object or explicit null.');
  }
  assertPackOperationFields(request.limits, ['maxInputBytes', 'maxOutputBytes', 'deadlineAt'], 'Pack operation limits');
  for (const key of ['maxInputBytes', 'maxOutputBytes', 'deadlineAt']) {
    if (!Number.isSafeInteger(request.limits[key]) || request.limits[key] <= 0) throw new Error(`Pack operation requires positive ${key}.`);
  }
  if (new TextEncoder().encode(JSON.stringify({ input: request.input, options: request.options })).length > request.limits.maxInputBytes) {
    throw new Error('Pack operation input exceeds maxInputBytes.');
  }
  return freezePackV2(request);
}
