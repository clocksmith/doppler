import { sha256BytesHex } from '../formats/sha256.js';
import { contractAssert as assert, count, hash, exactRecord, immutableRecord, metadataDigest } from '../config/canonical-contract-codec.js';

/** @type {import('./resident-block-contract.js').validateExecutionIdentity} */
export function validateExecutionIdentity(input) {
  const keys = ['schema', 'model', 'tokenizer', 'adapters', 'runtimeArchive', 'executionPlan',
    'partitionPlan', 'numericalPolicy', 'stateLayout', 'activationLayout', 'decoderLayout'];
  const value = /** @type {import('./resident-block-contract.js').ExecutionIdentity & Record<string, unknown>} */ (exactRecord(input, keys));
  assert(value.schema === 'doppler.execution-identity/v2', 'Unsupported execution identity.');
  assert(Array.isArray(value.adapters) && value.adapters.length <= 16, 'Adapter identity bound exceeded.');
  for (const adapter of value.adapters) hash(adapter, 'adapter');
  assert(new Set(value.adapters).size === value.adapters.length, 'Duplicate adapter identity.');
  for (const key of keys.filter(key => key !== 'schema' && key !== 'adapters')) hash(value[key], key);
  return immutableRecord(value);
}

/** @param {unknown} value @param {string} label */
function id(value, label) {
  assert(typeof value === 'string' && /^[A-Za-z0-9._:-]{1,256}$/.test(value), `Invalid ${label}.`);
}
/** @param {unknown} value @param {string} label */
function epoch(value, label) {
  assert(typeof value === 'string' && /^[a-f0-9]{32}$/.test(value), `Invalid ${label}.`);
}

/** @type {import('./resident-block-contract.js').validateResidentAttemptIdentity} */
export function validateResidentAttemptIdentity(input) {
  const value = /** @type {import('./resident-block-contract.js').ResidentAttemptIdentity} */ (exactRecord(input,
    ['schema', 'attemptId', 'requesterId', 'inputDigest', 'authorizationScope', 'routeId', 'generationId',
      'coordinatorEpoch', 'publisherEpoch', 'routeKind', 'stages']));
  assert(value.schema === 'doppler.resident-attempt/v2', 'Unsupported attempt identity.');
  id(value.attemptId, 'attempt'); id(value.requesterId, 'requester');
  epoch(value.coordinatorEpoch, 'coordinator epoch'); epoch(value.publisherEpoch, 'publisher epoch');
  for (const key of /** @type {const} */ (['inputDigest', 'authorizationScope', 'routeId', 'generationId'])) hash(value[key], key);
  assert(value.routeKind === 'whole-request' || value.routeKind === 'two-stage-layer', 'Unsupported route kind.');
  assert(Array.isArray(value.stages) && value.stages.length === (value.routeKind === 'whole-request' ? 1 : 2), 'Route cardinality mismatch.');
  for (let i = 0; i < value.stages.length; i++) {
    const stage = exactRecord(value.stages[i], ['stageId', 'participantId', 'participantEpoch', 'deviceEpoch', 'residentId', 'partitionIndex', 'executionId']);
    hash(stage.stageId, 'stage'); hash(stage.executionId, 'execution');
    id(stage.participantId, 'participant'); id(stage.residentId, 'resident');
    epoch(stage.participantEpoch, 'participant epoch'); epoch(stage.deviceEpoch, 'device epoch');
    assert(stage.partitionIndex === i, 'Stage order differs from partition order.');
  }
  assert(new Set(value.stages.map(stage => stage.stageId)).size === value.stages.length, 'Duplicate route stage.');
  return immutableRecord(value);
}

const activationKeys = ['schema', 'attemptDigest', 'sequence', 'baseCursorId', 'proposalId', 'sourceStageId',
  'destinationStageId', 'rows', 'tokenContextDigest', 'layout', 'payloadBytes', 'payloadDigest'];

/** @type {import('./resident-block-contract.js').validateBlockActivation} */
export function validateBlockActivation(input, bytes, limits) {
  const frame = /** @type {import('./resident-block-contract.js').BlockActivation} */ (exactRecord(input, activationKeys));
  assert(frame.schema === 'doppler.block-activation/v2', 'Unsupported activation frame.');
  hash(frame.baseCursorId, 'base cursor'); hash(frame.proposalId, 'proposal');
  const admittedBytes = validateActivationBytes(frame, bytes, limits);
  return Object.freeze({ frame: immutableRecord(frame), bytes: admittedBytes });
}

/** Ordinary bootstrap has no prior cursor and never invents a speculative proposal.
 * @type {import('./resident-block-contract.js').validateOrdinaryActivation} */
export function validateOrdinaryActivation(input, bytes, limits) {
  const frame = /** @type {import('./resident-block-contract.js').OrdinaryActivation} */ (exactRecord(input, activationKeys));
  assert(frame.schema === 'doppler.ordinary-activation/v1' && frame.proposalId === null, 'Unsupported ordinary activation frame.');
  if (frame.sequence === 0) {
    assert(frame.baseCursorId === null && frame.rows?.start === 0, 'Ordinary bootstrap must start at row zero without a cursor.');
  } else {
    hash(frame.baseCursorId, 'base cursor');
    assert(frame.rows?.start > 0 && frame.rows.end - frame.rows.start === 1, 'Ordinary decode consumes one absolute row.');
  }
  const admittedBytes = validateActivationBytes(frame, bytes, limits);
  return Object.freeze({ frame: immutableRecord(frame), bytes: admittedBytes });
}

/** @param {import('./resident-block-contract.js').BlockActivation | import('./resident-block-contract.js').OrdinaryActivation} frame
 * @param {Uint8Array} bytes @param {{ maxRows: number, maxBytes: number }} limits */
function validateActivationBytes(frame, bytes, limits) {
  for (const key of /** @type {const} */ (['attemptDigest', 'sourceStageId', 'destinationStageId', 'tokenContextDigest', 'payloadDigest'])) hash(frame[key], key);
  assert(frame.sourceStageId !== frame.destinationStageId, 'Activation source and destination must differ.');
  count(frame.sequence, 'sequence');
  exactRecord(frame.rows, ['start', 'end']);
  count(frame.rows.start, 'row start'); count(frame.rows.end, 'row end');
  count(limits.maxRows, 'row bound'); count(limits.maxBytes, 'payload bound');
  const rows = frame.rows.end - frame.rows.start;
  assert(rows > 0 && rows <= limits.maxRows, 'Activation row bound exceeded.');
  const layout = exactRecord(frame.layout, ['dtype', 'precision', 'components', 'elementStrideBytes', 'hiddenSize', 'byteOrder']);
  count(layout.hiddenSize, 'hidden size');
  assert(layout.hiddenSize > 0 && layout.byteOrder === 'little-endian', 'Invalid activation geometry.');
  assert(layout.precision === 'two-f32'
    ? layout.dtype === 'f32' && layout.components === 2 && layout.elementStrideBytes === 8
    : (layout.dtype === 'f16' || layout.dtype === 'f32') && layout.precision === layout.dtype
      && layout.components === 1 && layout.elementStrideBytes === (layout.dtype === 'f16' ? 2 : 4),
  'Activation layout must explicitly declare matching components and stride.');
  const expected = rows * layout.hiddenSize * layout.elementStrideBytes;
  count(expected, 'activation byte product', limits.maxBytes);
  assert(bytes instanceof Uint8Array && bytes.byteLength === expected && frame.payloadBytes === expected, 'Activation length mismatch.');
  assert(`sha256:${sha256BytesHex(bytes)}` === frame.payloadDigest, 'Activation payload digest mismatch.');
  return bytes.slice();
}

/** @type {import('./resident-block-contract.js').residentAttemptDigest} */
export function residentAttemptDigest(input) {
  const identity = validateResidentAttemptIdentity(input);
  return metadataDigest(identity.schema, identity);
}
