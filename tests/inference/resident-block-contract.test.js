import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateExecutionIdentity, validateResidentAttemptIdentity, residentAttemptDigest, validateBlockActivation } from '../../src/inference/resident-block-contract.js';
const hash = 'sha256:' + 'a'.repeat(64), other = 'sha256:' + 'b'.repeat(64), epoch = '0'.repeat(32);
const execution = { schema: 'doppler.execution-identity/v2', model: hash, tokenizer: hash, adapters: [], runtimeArchive: hash,
  executionPlan: hash, partitionPlan: hash, numericalPolicy: hash, stateLayout: hash, activationLayout: hash, decoderLayout: hash };
assert(Object.isFrozen(validateExecutionIdentity(execution)));
assert.throws(() => validateExecutionIdentity({ ...execution, archive: hash }));
const stage = { stageId: hash, participantId: 'p', participantEpoch: epoch, deviceEpoch: epoch, residentId: 'r', partitionIndex: 0, executionId: hash };
const attempt = { schema: 'doppler.resident-attempt/v2', attemptId: 'a', requesterId: 'q', inputDigest: hash, authorizationScope: hash,
  routeId: hash, generationId: hash, coordinatorEpoch: epoch, publisherEpoch: epoch, routeKind: 'whole-request', stages: [stage] };
assert.equal(validateResidentAttemptIdentity(attempt).stages.length, 1);
assert.throws(() => validateResidentAttemptIdentity({ ...attempt, stages: [stage, stage] }));
assert.throws(() => validateResidentAttemptIdentity({ ...attempt, publisherEpoch: '' }));
assert.throws(() => validateResidentAttemptIdentity({ ...attempt, stages: [{ ...stage, deviceEpoch: undefined }] }));
const split = { ...attempt, routeKind: 'two-stage-layer', stages: [stage, { ...stage, stageId: other, partitionIndex: 1 }] };
assert.equal(validateResidentAttemptIdentity(split).stages.length, 2);
for (const [dtype, precision, components, stride] of [['f16', 'f16', 1, 2], ['f32', 'f32', 1, 4], ['f32', 'two-f32', 2, 8]]) {
  const bytes = new Uint8Array(3 * 4 * stride);
  const frame = { schema: 'doppler.block-activation/v2', attemptDigest: residentAttemptDigest(split), sequence: 1,
    baseCursorId: hash, proposalId: hash, sourceStageId: hash, destinationStageId: other, rows: { start: 7, end: 10 },
    tokenContextDigest: hash, layout: {dtype, precision, components, elementStrideBytes: stride, hiddenSize: 4, byteOrder: 'little-endian'},
    payloadBytes: bytes.length, payloadDigest: 'sha256:' + createHash('sha256').update(bytes).digest('hex') };
  const admitted = validateBlockActivation(frame, bytes, {maxRows: 5, maxBytes: 1024});
  bytes[0] = 1; assert.equal(admitted.bytes[0], 0);
  assert.throws(() => validateBlockActivation(frame, bytes, {maxRows: 5, maxBytes: 1024}), /digest/);
  bytes[0] = 0;
  assert.throws(() => validateBlockActivation(frame, new Uint8Array(bytes.length + 1), {maxRows: 5, maxBytes: 1024}), /length/);
  assert.throws(() => validateBlockActivation(frame, bytes, {maxRows: 2, maxBytes: 1024}), /row bound/);
  assert.throws(() => validateBlockActivation({...frame, layout: {...frame.layout, components: undefined}}, bytes, {maxRows: 5, maxBytes: 1024}), /layout/);
  assert.throws(() => validateBlockActivation({...frame, rows: {start: 0, end: Number.MAX_SAFE_INTEGER}}, bytes,
    {maxRows: Number.MAX_SAFE_INTEGER, maxBytes: Number.MAX_SAFE_INTEGER}));
}
console.log('resident-block-contract: complete identities, route cardinality, explicit layouts and exact payload bounds passed');
