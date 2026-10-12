import { contractAssert as assert, exactRecord, canonicalMetadata, metadataDigest, hash, count,
  scalarString, admitTokens, immutableRecord } from '../config/canonical-contract-codec.js';
import { validateResidentAttemptIdentity, residentAttemptDigest } from './resident-block-contract.js';
import { validateResidentCursor } from './resident-state-contract.js';
import { GENERATION_CONTRACT } from '../config/generation-contract.js';

/** @type {import('./resident-publication-contract.js').residentPublicationDecisionDigest} */
export function residentPublicationDecisionDigest(input) {
  return metadataDigest('doppler.resident-decision/v1', { attemptDigest: input.attemptDigest,
    sequence: input.sequence, baseCursorId: input.baseCursorId, acceptedTokenIds: input.acceptedTokenIds,
    tokenRange: input.tokenRange, successorCursorId: input.successor.cursorId, delta: input.delta,
    stoppingOutcome: input.stoppingOutcome });
}

/** @type {import('./resident-publication-contract.js').validateExecutionPublication} */
export function validateExecutionPublication(input, attempt, policy, vocabSize, base) {
  // Enforce the complete control-message bound before copying arrays or payloads.
  canonicalMetadata(input);
  const value = /** @type {import('./resident-publication-contract.js').ExecutionPublication} */ (exactRecord(input,
    ['schema', 'publicationDigest', 'attemptId', 'coordinatorEpoch', 'attemptDigest', 'sequence', 'expectedRevision',
      'previousPublicationDigest', 'baseCursorId', 'decisionId', 'acceptedTokenIds', 'tokenRange', 'successor',
      'delta', 'stoppingOutcome', 'acknowledgments']));
  attempt = validateResidentAttemptIdentity(attempt);
  const attemptDigest = residentAttemptDigest(attempt);
  assert(value.schema === 'doppler.execution-publication/v2' && value.attemptDigest === attemptDigest
    && value.attemptId === attempt.attemptId && value.coordinatorEpoch === attempt.coordinatorEpoch,
  'Publication attempt binding mismatch.');
  count(value.sequence, 'publication sequence');
  if (value.previousPublicationDigest !== null) hash(value.previousPublicationDigest, 'previous publication');
  const successor = validateResidentCursor(value.successor, policy, vocabSize);
  assert(successor.generationId === attempt.generationId, 'Publication generation mismatch.');
  const ids = admitTokens(value.acceptedTokenIds, vocabSize, GENERATION_CONTRACT.canonical.maxProposalTokens + 1);
  assert(ids.length > 0, 'Execution publication requires accepted tokens.');
  exactRecord(value.tokenRange, ['start', 'end']);
  count(value.tokenRange.start, 'range start'); count(value.tokenRange.end, 'range end');
  assert(value.tokenRange.end - value.tokenRange.start === ids.length
    && value.tokenRange.end === successor.acceptedGeneratedTokenCount
    && ids.at(-1) === successor.pendingTokenId, 'Publication accepted range mismatch.');
  if (base === null) {
    assert(value.sequence === 0 && value.expectedRevision === null && value.baseCursorId === null
      && successor.revision === 0 && value.tokenRange.start === 0 && ids.length === 1,
    'Publication bootstrap mismatch.');
  } else {
    base = validateResidentCursor(base, policy, vocabSize);
    assert(!base.terminal && base.generationId === attempt.generationId && value.sequence > 0
      && value.expectedRevision === base.revision && value.baseCursorId === base.cursorId
      && successor.revision === base.revision + 1 && value.tokenRange.start === base.acceptedGeneratedTokenCount
      && successor.promptTokenCount === base.promptTokenCount
      && successor.materializedTokenCount === base.materializedTokenCount + ids.length,
    'Publication successor does not extend its base cursor.');
  }
  exactRecord(value.delta, ['text', 'outputDigest']);
  scalarString(value.delta.text); hash(value.delta.outputDigest, 'output state digest');
  assert(value.delta.text.length <= policy.maxVisibleOutputUtf16, 'Publication delta exceeds visible output allowance.');
  assert(value.stoppingOutcome === successor.stopReason, 'Publication stopping outcome mismatch.');
  assert(value.decisionId === residentPublicationDecisionDigest(value), 'Publication decision digest mismatch.');
  assert(Array.isArray(value.acknowledgments) && value.acknowledgments.length === attempt.stages.length,
    'Publication acknowledgment set must equal its route.');
  const seen = new Set();
  for (const ack of value.acknowledgments) {
    exactRecord(ack, ['schema', 'stageId', 'attemptDigest', 'sequence', 'baseCursorId', 'decisionId',
      'successorCursorId', 'participantEpoch', 'deviceEpoch', 'residentId', 'authenticationDigest']);
    const stage = attempt.stages.find(item => item.stageId === ack.stageId);
    assert(stage && !seen.has(ack.stageId), 'Duplicate or extra publication acknowledgment.');
    seen.add(ack.stageId);
    assert(ack.schema === 'doppler.stage-commit/v1' && ack.attemptDigest === attemptDigest
      && ack.sequence === value.sequence && ack.baseCursorId === value.baseCursorId
      && ack.decisionId === value.decisionId && ack.successorCursorId === successor.cursorId
      && ack.participantEpoch === stage.participantEpoch && ack.deviceEpoch === stage.deviceEpoch
      && ack.residentId === stage.residentId, 'Publication acknowledgment binding mismatch.');
    hash(ack.authenticationDigest, 'authenticated acknowledgment evidence');
  }
  const { publicationDigest, ...body } = value;
  assert(publicationDigest === metadataDigest(body.schema, body), 'Publication digest mismatch.');
  return immutableRecord(value);
}
