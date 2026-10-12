import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bootstrapResidentCursor, decideResidentBlock } from '../../src/inference/resident-state-contract.js';
import { residentAttemptDigest } from '../../src/inference/resident-block-contract.js';
import { metadataDigest } from '../../src/config/canonical-contract-codec.js';
import { residentPublicationDecisionDigest, validateExecutionPublication } from '../../src/inference/resident-publication-contract.js';
const policy = JSON.parse(readFileSync(new URL('../fixtures/canonical-generation-policy.json', import.meta.url)));
const hash = 'sha256:' + 'a'.repeat(64), other = 'sha256:' + 'b'.repeat(64), epoch = '0'.repeat(32);
const stage = { stageId: hash, participantId: 'p', participantEpoch: epoch, deviceEpoch: epoch, residentId: 'r', partitionIndex: 0, executionId: hash };
const whole = { schema: 'doppler.resident-attempt/v2', attemptId: 'a', requesterId: 'q', inputDigest: hash, authorizationScope: hash,
  routeId: hash, generationId: hash, coordinatorEpoch: epoch, publisherEpoch: epoch, routeKind: 'whole-request', stages: [stage] };
const split = { ...whole, routeKind: 'two-stage-layer', stages: [stage, { ...stage, stageId: other, partitionIndex: 1 }] };
const bootstrap = bootstrapResidentCursor([1, 2], 3, policy, 100, hash);
const seal = value => {
  const { publicationDigest, ...body } = value;
  return { ...body, publicationDigest: metadataDigest(body.schema, body) };
};
const publication = (attempt, base = null) => {
  const next = base ? decideResidentBlock(base, [1, 2], [], [4], [false], policy, 100) : {
    acceptedTokenIds: [3], tokenRange: { start: 0, end: 1 }, successor: bootstrap };
  const value = { schema: 'doppler.execution-publication/v2', attemptId: attempt.attemptId,
    coordinatorEpoch: epoch, attemptDigest: residentAttemptDigest(attempt), sequence: base ? 1 : 0,
    expectedRevision: base?.revision ?? null, baseCursorId: base?.cursorId ?? null, previousPublicationDigest: base ? hash : null,
    acceptedTokenIds: next.acceptedTokenIds, tokenRange: next.tokenRange, successor: next.successor,
    delta: { text: 'x', outputDigest: hash }, stoppingOutcome: next.successor.stopReason };
  value.decisionId = residentPublicationDecisionDigest(value);
  value.acknowledgments = attempt.stages.map(item => ({ schema: 'doppler.stage-commit/v1', stageId: item.stageId,
    attemptDigest: value.attemptDigest, sequence: value.sequence, baseCursorId: value.baseCursorId,
    decisionId: value.decisionId, successorCursorId: value.successor.cursorId, participantEpoch: item.participantEpoch,
    deviceEpoch: item.deviceEpoch, residentId: item.residentId, authenticationDigest: hash }));
  return seal(value);
};
for (const attempt of [whole, split]) {
  for (const base of [null, bootstrap]) {
    const value = publication(attempt, base);
    assert(Object.isFrozen(validateExecutionPublication(value, attempt, policy, 100, base)));
    for (const mutate of [
      v => { v.acknowledgments = []; },
      v => { v.acknowledgments.push(v.acknowledgments[0]); },
      v => { v.acknowledgments[0].deviceEpoch = '1'.repeat(32); },
      v => { v.acknowledgments[0].decisionId = other; },
      v => { v.acknowledgments[0].successorCursorId = other; },
      v => { v.acknowledgments[0].authenticationDigest = ''; },
      v => { v.acknowledgments[0].stageId = 'sha256:' + 'f'.repeat(64); },
      v => { v.delta.text = 'changed'; },
      v => { v.tokenRange.start++; },
      v => { v.acceptedTokenIds[0] = 50; },
      v => { v.coordinatorEpoch = '1'.repeat(32); },
      v => { v.expectedRevision = 400; },
      v => { v.stoppingOutcome = 'max-tokens'; },
      v => { v.extra = true; },
    ]) {
      const changed = structuredClone(value); mutate(changed);
      assert.throws(() => validateExecutionPublication(seal(changed), attempt, policy, 100, base));
    }
    if (attempt === split) {
      const changed = structuredClone(value); changed.acknowledgments[1] = changed.acknowledgments[0];
      assert.throws(() => validateExecutionPublication(seal(changed), attempt, policy, 100, base), /Duplicate/);
    }
  }
}
console.log('publication contracts: exact one/two-stage acknowledgment sets, epochs, decisions, cursor and contiguous ranges passed');
