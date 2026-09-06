import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { getCapsuleIdentity } from '../../src/config/capsule.js';
import { hashTargetPlan } from '../../src/config/target-plan.js';
import { qualifySequenceCapsule, validateSequenceCapsuleQualificationConfig } from '../../tools/qualify-sequence-capsule.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

const fixture = await createSignedCapsuleFixture();
const directory = await mkdtemp(join(tmpdir(), 'doppler-sequence-capsule-contract-'));
const capsulePath = join(directory, 'capsule.json');
const referencePath = join(directory, 'reference.json');
await writeFile(capsulePath, JSON.stringify(fixture.capsule));
await writeFile(referencePath, '{}');
const config = {
  capsulePath, referencePath, outputPath: join(directory, 'failure.json'),
  referenceDigest: `sha256:${'a'.repeat(64)}`,
  expectedCapsule: getCapsuleIdentity(fixture.capsule),
  originPolicy: 'disabled',
  openOptions: {
    trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
    acceptedTargetPlanDigests: fixture.capsule.targetPlans.map(hashTargetPlan),
  },
  sequenceOptions: { includeTokenEmbeddings: true, includeLogits: false, assignment: { id: 'contract-only', attempt: 1 } },
};
assert.equal(validateSequenceCapsuleQualificationConfig(config), config);
for (const [name, mutate] of [
  ['missing path', (value) => { delete value.capsulePath; }],
  ['missing reference', (value) => { delete value.referenceDigest; }],
  ['missing Capsule', (value) => { delete value.expectedCapsule; }],
  ['partial Capsule', (value) => { delete value.expectedCapsule.envelopeDigest; }],
  ['unknown Capsule', (value) => { value.expectedCapsule.schema = 'unknown'; }],
  ['enabled origin', (value) => { value.originPolicy = 'enabled'; }],
  ['missing trust', (value) => { value.openOptions.trustedSigners = {}; }],
  ['malformed trust', (value) => { value.openOptions.trustedSigners = 'signer'; }],
  ['missing plans', (value) => { value.openOptions.acceptedTargetPlanDigests = []; }],
  ['missing assignment', (value) => { delete value.sequenceOptions.assignment; }],
  ['wrong operation options', (value) => { value.sequenceOptions.includeLogits = true; }],
]) {
  const invalid = structuredClone(config);
  mutate(invalid);
  assert.throws(() => validateSequenceCapsuleQualificationConfig(invalid), /Qualification/, name);
}

// Retained adverse observations must survive pre-GPU failures; no model is injected.
const originalFetch = globalThis.fetch;
const changedCapsule = structuredClone(config);
changedCapsule.expectedCapsule.envelopeDigest = `sha256:${'0'.repeat(64)}`;
const capsuleFailure = await qualifySequenceCapsule(changedCapsule);
assert.equal(capsuleFailure.passed, false);
assert.equal(capsuleFailure.stage, 'input-verification');
assert.match(capsuleFailure.error.message, /Capsule identity differs/);
assert.deepEqual(JSON.parse(await readFile(config.outputPath, 'utf8')), capsuleFailure);
assert.equal(globalThis.fetch, originalFetch);
const referenceFailure = await qualifySequenceCapsule({ ...config, outputPath: join(directory, 'reference-failure.json') });
assert.equal(referenceFailure.passed, false);
assert.match(referenceFailure.error.message, /Reference digest differs/);
assert.equal(referenceFailure.runtime, undefined);
assert.equal(globalThis.fetch, originalFetch);
console.log('qualify-sequence-capsule.test: ok');
