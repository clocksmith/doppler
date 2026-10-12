import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fixture from '../fixtures/canonical-generation-policy.json' with { type: 'json' };
import { canonicalMetadata, metadataDigest, tokenDigest } from '../../src/config/canonical-contract-codec.js';
import { validateCanonicalGenerationPolicy, admitCanonicalGenerationPolicy, canonicalGenerationIdentity } from '../../src/config/canonical-generation-policy.js';
import { admitResidentPrompt, bootstrapResidentCursor, validateResidentCursor, admitResidentBlock, decideResidentBlock } from '../../src/inference/resident-state-contract.js';

const policy = validateCanonicalGenerationPolicy(fixture, 100);
const generation = canonicalGenerationIdentity(policy, `sha256:${'c'.repeat(64)}`, 100).generationId;
const prompt = [1, 2, 3, 4, 5];
const bootstrap = bootstrapResidentCursor(prompt, 6, policy, 100, generation);
const first = decideResidentBlock(bootstrap, prompt, [], [7], [false], policy, 100);
const base = first.successor, history = [...prompt, 6];
assert.equal(base.materializedTokenCount, 6);
assert.equal(base.acceptedGeneratedTokenCount, 2);
assert.equal(base.pendingTokenId, 7);
for (const [proposal, targets, output, materialized, generated] of [
  [[], [8], [8], 7, 3], [[10, 11, 12], [8], [8], 7, 3],
  [[10, 11, 12], [10, 8], [10, 8], 8, 4],
  [[10, 11, 12], [10, 11, 12, 8], [10, 11, 12, 8], 10, 6],
  [[10, 99, 12], [10, 99], [10, 99], 8, 4],
]) {
  const result = decideResidentBlock(base, history, proposal, targets, targets.map(() => false), policy, 100);
  assert.deepEqual(result.acceptedTokenIds, output);
  assert.equal(result.successor.materializedTokenCount, materialized);
  assert.equal(result.successor.acceptedGeneratedTokenCount, generated);
  assert.equal(result.successor.revision, base.revision + 1);
  assert.equal(result.successor.pendingTokenId, output.at(-1));
  assert.equal(result.successor.terminal, output.at(-1) === 99);
}
for (let k = 0; k <= 4; k++) {
  const proposal = Array.from({ length: k }, (_, i) => 10 + i);
  for (let position = 0; position <= k; position++) {
    for (const last of [50, 99, 98, 97]) {
      const choices = [...proposal.slice(0, position), last];
      const result = decideResidentBlock(base, history, proposal, choices, choices.map(() => false), policy, 100);
      assert.deepEqual(result.acceptedTokenIds, choices);
      assert.equal(result.materializeInputCount, position + 1);
      assert.equal(result.successor.materializedTokenCount, history.length + position + 1);
      assert.equal(result.successor.acceptedTokenDigest,
        tokenDigest([...history, base.pendingTokenId, ...choices], 100, 65));
    }
  }
}
assert.throws(() => decideResidentBlock(base, history, [10, 11], [10], [false], policy, 100), /Missing correction/);
assert.throws(() => decideResidentBlock(base, [...history].reverse(), [], [8], [false], policy, 100), /history mismatch/);
assert.throws(() => admitResidentBlock(base, Array(5).fill(8), policy, 100));
assert.throws(() => validateResidentCursor({ ...base, materializedTokenCount: 8 }, policy, 100));
assert.throws(() => validateResidentCursor({ ...base, surprise: 1 }, policy, 100));
for (const change of [{ seedHex64: '0' }, { seedHex64: 0 }, { schema: 'doppler.generation-policy/v2' },
  { temperature: Infinity }, { topK: 0.5 }, { presencePenaltyWindow: -1 }, { suppressTokenIds: [100] },
  { eosTokenIds: [99, 99] }, { stopSequences: ['\ud800'] }, { stopSequences: ['a'.repeat(257)] },
  { maxGeneratedTokens: Number.MAX_SAFE_INTEGER + 1 }, { toolBoundaryPolicy: 'ignore' }, { unknown: 0 }]) {
  assert.throws(() => validateCanonicalGenerationPolicy({ ...fixture, ...change }, 100));
}
assert(Object.isFrozen(policy.eosTokenIds));
const omitted = { ...fixture }; delete omitted.seedHex64;
let persisted;
const generated = await admitCanonicalGenerationPolicy(omitted, 100, async value => { persisted = value; });
assert.equal(generated, persisted);
assert.match(generated.seedHex64, /^[a-f0-9]{16}$/);
await assert.rejects(admitCanonicalGenerationPolicy(omitted, 100, async () => { throw Error('storage'); }), /storage/);
for (const [change, reason] of [[{ maxGeneratedTokens: 0 }, 'max-tokens'], [{ totalAcceptedTokenLimit: 5 }, 'context-limit']]) {
  const options = { ...policy, ...change };
  const admitted = admitResidentPrompt(prompt, options, 100, generation);
  assert.equal(admitted.schema, 'doppler.admission-only/v1');
  assert.equal(admitted.stopReason, reason);
  assert(!('cursor' in admitted));
  assert.throws(() => bootstrapResidentCursor(prompt, 8, options, 100, generation));
}
for (const token of [8, 99]) {
  const options = { ...policy, materializedInputCapacity: 5 };
  const cursor = bootstrapResidentCursor(prompt, token, options, 100, generation);
  assert.equal(cursor.stopReason, token === 99 ? 'eos-token' : 'context-limit');
  assert.throws(() => admitResidentBlock(cursor, [], options, 100));
}
assert.throws(() => admitResidentPrompt([], policy, 100, generation));
const oneLeft = { ...policy, maxGeneratedTokens: 3 };
assert.deepEqual(admitResidentBlock(base, [], oneLeft, 100), [7]);
assert.throws(() => admitResidentBlock(base, [8], oneLeft, 100));
assert.equal(decideResidentBlock(base, history, [], [8], [false], oneLeft, 100).successor.stopReason, 'max-tokens');
assert.equal(canonicalMetadata({ z: -0, a: [1e30, '\u20ac'] }), '{"a":[1e+30,"€"],"z":0}');
for (const value of [NaN, Infinity, undefined, '\udfff', new Date(), [,,], { get x() { throw Error('must not run'); } }, 'x'.repeat(65537)]) {
  assert.throws(() => canonicalMetadata(value));
}
const cyclic = {}; cyclic.self = cyclic; assert.throws(() => canonicalMetadata(cyclic));
const bytes = Buffer.alloc(8 + 3 * 4); bytes.writeBigUInt64LE(3n);
[0, 1, 99].forEach((value, i) => bytes.writeUInt32LE(value, 8 + i * 4));
assert.equal(tokenDigest([0, 1, 99], 100, 3), 'sha256:' + createHash('sha256').update('doppler.tokens/v1\0').update(bytes).digest('hex'));
assert.equal(metadataDigest('test/v1', { b: 2, a: 1 }), 'sha256:' + createHash('sha256').update('test/v1\0{"a":1,"b":2}').digest('hex'));
console.log('resident-state-contract: bounded encoding, policy admission, cursor and all replay positions passed');
