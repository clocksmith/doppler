import { GENERATION_CONTRACT } from '../config/generation-contract.js';
import { validateCanonicalGenerationPolicy } from '../config/canonical-generation-policy.js';
import { contractAssert as assert, count, hash, exactRecord, immutableRecord, metadataDigest,
  admitTokens, tokenDigest } from '../config/canonical-contract-codec.js';

const cursorFields = ['schema', 'cursorId', 'generationId', 'revision', 'promptTokenCount',
  'materializedTokenCount', 'acceptedGeneratedTokenCount', 'nextTargetTokenIndex', 'pendingTokenId',
  'materializedPrefixDigest', 'acceptedTokenDigest', 'terminal', 'stopReason'];

/** @type {import('./resident-state-contract.js').canonicalStopReason} */
export function canonicalStopReason(policy, tokenId, generatedCount, promptCount, materializedCount, textStop = false) {
  if (policy.eosTokenIds.includes(tokenId)) return 'eos-token';
  if (policy.stopTokenIds.includes(tokenId)) return 'stop-token';
  if (policy.toolTokenIds.includes(tokenId)) {
    assert(policy.toolBoundaryPolicy === 'stop', 'Tool boundary rejected by generation policy.');
    return 'tool-boundary';
  }
  if (textStop) return 'stop-sequence';
  if (generatedCount >= policy.maxGeneratedTokens) return 'max-tokens';
  if (materializedCount >= policy.materializedInputCapacity
    || generatedCount >= policy.totalAcceptedTokenLimit - promptCount) return 'context-limit';
  return null;
}

/** @type {import('./resident-state-contract.js').validateResidentCursor} */
export function validateResidentCursor(input, policy, vocabSize) {
  policy = validateCanonicalGenerationPolicy(policy, vocabSize);
  const cursor = /** @type {import('./resident-state-contract.js').ResidentCursor & Record<string, unknown>} */ (exactRecord(input, cursorFields));
  assert(cursor.schema === 'doppler.cursor/v2', 'Unsupported cursor schema.');
  for (const key of ['revision', 'promptTokenCount', 'materializedTokenCount', 'acceptedGeneratedTokenCount', 'nextTargetTokenIndex']) count(cursor[key], key);
  for (const key of ['cursorId', 'generationId', 'materializedPrefixDigest', 'acceptedTokenDigest']) hash(cursor[key], key);
  if (cursor.pendingTokenId !== null) count(cursor.pendingTokenId, 'pending token', vocabSize - 1);
  const p = cursor.promptTokenCount, g = cursor.acceptedGeneratedTokenCount, m = cursor.materializedTokenCount;
  assert(p <= policy.materializedInputCapacity && p <= policy.totalAcceptedTokenLimit
    && g <= policy.maxGeneratedTokens && g <= policy.totalAcceptedTokenLimit - p
    && m <= policy.materializedInputCapacity, 'Cursor exceeds generation bounds.');
  assert(g > 0 && cursor.pendingTokenId !== null, 'Reusable cursor requires an accepted pending token.');
  assert(m === p + g - 1 && cursor.nextTargetTokenIndex === g, 'Cursor materialization invariant failed.');
  assert(typeof cursor.terminal === 'boolean'
    && (cursor.stopReason === null || GENERATION_CONTRACT.canonical.stopReasons.includes(cursor.stopReason))
    && cursor.terminal === (cursor.stopReason !== null), 'Invalid cursor terminal state.');
  const reason = canonicalStopReason(policy, cursor.pendingTokenId, g, p, m, cursor.stopReason === 'stop-sequence');
  assert(cursor.stopReason === reason, 'Cursor stopping outcome differs from policy.');
  const { cursorId, ...body } = cursor;
  assert(cursorId === metadataDigest(body.schema, body), 'Cursor digest mismatch.');
  return immutableRecord(cursor);
}

/** @param {Pick<import('./resident-state-contract.js').ResidentCursor, 'generationId'|'revision'|'promptTokenCount'|'acceptedGeneratedTokenCount'|'pendingTokenId'> & {materialized: readonly number[]; textStop: boolean}} input
 * @param {import('../config/generation-contract.js').CanonicalGenerationPolicy} policy @param {number} vocabSize */
function makeCursor({ generationId, revision, promptTokenCount, acceptedGeneratedTokenCount,
  materialized, pendingTokenId, textStop }, policy, vocabSize) {
  const stopReason = canonicalStopReason(policy, pendingTokenId, acceptedGeneratedTokenCount,
    promptTokenCount, materialized.length, textStop);
  const body = { schema: 'doppler.cursor/v2', generationId, revision, promptTokenCount,
    materializedTokenCount: materialized.length, acceptedGeneratedTokenCount,
    nextTargetTokenIndex: acceptedGeneratedTokenCount, pendingTokenId,
    materializedPrefixDigest: tokenDigest(materialized, vocabSize, policy.materializedInputCapacity),
    acceptedTokenDigest: tokenDigest([...materialized, pendingTokenId], vocabSize, policy.totalAcceptedTokenLimit),
    terminal: stopReason !== null, stopReason };
  return validateResidentCursor({ ...body, cursorId: metadataDigest(body.schema, body) }, policy, vocabSize);
}

/** @type {import('./resident-state-contract.js').admitResidentPrompt} */
export function admitResidentPrompt(prompt, policy, vocabSize, generationId) {
  policy = validateCanonicalGenerationPolicy(policy, vocabSize);
  hash(generationId, 'generation identity');
  const tokens = admitTokens(prompt, vocabSize, Math.min(policy.materializedInputCapacity, policy.totalAcceptedTokenLimit));
  // This is the tokenizer's effective prompt; an empty effective prompt needs
  // an independently declared tokenizer rule and is not inferred here.
  assert(tokens.length > 0, 'Effective tokenizer prompt must contain at least one token.');
  if (policy.maxGeneratedTokens === 0 || tokens.length === policy.totalAcceptedTokenLimit) {
    return immutableRecord({ schema: 'doppler.admission-only/v1', generationId,
      promptTokenCount: tokens.length, promptDigest: tokenDigest(tokens, vocabSize, policy.materializedInputCapacity),
      acceptedTokenIds: [], stopReason: policy.maxGeneratedTokens === 0 ? 'max-tokens' : 'context-limit' });
  }
  return Object.freeze({ kind: 'prefill', tokenIds: tokens, generationId });
}

/** @type {import('./resident-state-contract.js').bootstrapResidentCursor} */
export function bootstrapResidentCursor(prompt, firstToken, policy, vocabSize, generationId, textStop = false) {
  const admitted = admitResidentPrompt(prompt, policy, vocabSize, generationId);
  assert('kind' in admitted && admitted.kind === 'prefill', 'Admission-only result cannot create an inference cursor.');
  count(firstToken, 'first token', vocabSize - 1);
  return makeCursor({ generationId, revision: 0, promptTokenCount: prompt.length, acceptedGeneratedTokenCount: 1,
    materialized: admitted.tokenIds, pendingTokenId: firstToken, textStop }, policy, vocabSize);
}

/** @type {import('./resident-state-contract.js').admitResidentBlock} */
export function admitResidentBlock(cursor, proposal, policy, vocabSize) {
  cursor = validateResidentCursor(cursor, policy, vocabSize);
  assert(!cursor.terminal, 'Terminal cursor cannot start a block.');
  const remaining = policy.maxGeneratedTokens - cursor.acceptedGeneratedTokenCount;
  assert(remaining > 0, 'No remaining output allowance.');
  const tokens = admitTokens(proposal, vocabSize, Math.min(GENERATION_CONTRACT.canonical.maxProposalTokens, remaining - 1));
  const rows = tokens.length + 1;
  assert(rows <= policy.materializedInputCapacity - cursor.materializedTokenCount
    && rows <= policy.totalAcceptedTokenLimit - cursor.promptTokenCount - cursor.acceptedGeneratedTokenCount,
  'Block exceeds materialized or accepted context bounds.');
  return Object.freeze([cursor.pendingTokenId, ...tokens]);
}

// Target choices are produced once by the final-stage sampler. This pure
// transition never samples, decodes, publishes, or owns numerical state.
/** @type {import('./resident-state-contract.js').decideResidentBlock} */
export function decideResidentBlock(cursor, materialized, proposal, targets, textStops, policy, vocabSize) {
  const inputs = admitResidentBlock(cursor, proposal, policy, vocabSize);
  const history = admitTokens(materialized, vocabSize, policy.materializedInputCapacity);
  assert(history.length === cursor.materializedTokenCount
    && tokenDigest(history, vocabSize, policy.materializedInputCapacity) === cursor.materializedPrefixDigest
    && tokenDigest([...history, cursor.pendingTokenId], vocabSize, policy.totalAcceptedTokenLimit) === cursor.acceptedTokenDigest,
  'Cursor token history mismatch.');
  const choices = admitTokens(targets, vocabSize, inputs.length);
  assert(choices.length > 0 && Array.isArray(textStops) && textStops.length === choices.length
    && textStops.every(value => typeof value === 'boolean'), 'Invalid target stopping outcomes.');
  const accepted = [];
  let matches = 0, stopReason = null;
  for (let j = 0; j < choices.length; j++) {
    const token = choices[j];
    accepted.push(token);
    stopReason = canonicalStopReason(policy, token, cursor.acceptedGeneratedTokenCount + accepted.length,
      cursor.promptTokenCount, cursor.materializedTokenCount + accepted.length, textStops[j]);
    if (j < proposal.length && token === proposal[j]) matches++;
    if (stopReason || j === proposal.length || token !== proposal[j]) break;
  }
  assert(stopReason || accepted.length === inputs.length || matches < accepted.length, 'Missing correction or bonus target.');
  assert(cursor.revision < Number.MAX_SAFE_INTEGER, 'Cursor revision overflow.');
  const nextHistory = [...history, ...inputs.slice(0, accepted.length)];
  const successor = makeCursor({ generationId: cursor.generationId, revision: cursor.revision + 1,
    promptTokenCount: cursor.promptTokenCount, acceptedGeneratedTokenCount: cursor.acceptedGeneratedTokenCount + accepted.length,
    materialized: nextHistory, pendingTokenId: accepted[accepted.length - 1], textStop: stopReason === 'stop-sequence' }, policy, vocabSize);
  return immutableRecord({ acceptedTokenIds: accepted, acceptedProposalCount: matches,
    materializeInputCount: accepted.length, tokenRange: { start: cursor.acceptedGeneratedTokenCount,
      end: successor.acceptedGeneratedTokenCount }, successor });
}
