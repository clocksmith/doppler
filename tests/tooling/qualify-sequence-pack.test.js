import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { getPackIdentity } from '../../src/config/pack.js';
import { hashTargetPlan } from '../../src/config/target-plan.js';
import { qualifySequencePack, validateSequencePackQualificationConfig } from '../../tools/qualify-sequence-pack.js';
import { createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';

const fixture = await createSignedPackFixture();
const directory = await mkdtemp(join(tmpdir(), 'doppler-sequence-pack-contract-'));
const packPath = join(directory, 'pack.json');
const referencePath = join(directory, 'reference.json');
await writeFile(packPath, JSON.stringify(fixture.pack));
await writeFile(referencePath, '{}');
const config = {
  packPath, referencePath, outputPath: join(directory, 'failure.json'),
  referenceDigest: `sha256:${'a'.repeat(64)}`,
  expectedPack: getPackIdentity(fixture.pack),
  originPolicy: 'disabled',
  openOptions: {
    trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
    acceptedTargetPlanDigests: fixture.pack.targetPlans.map(hashTargetPlan),
  },
  sequenceOptions: { includeTokenEmbeddings: true, includeLogits: false, assignment: { id: 'contract-only', attempt: 1 } },
};
assert.equal(validateSequencePackQualificationConfig(config), config);
for (const [name, mutate] of [
  ['missing path', (value) => { delete value.packPath; }],
  ['missing reference', (value) => { delete value.referenceDigest; }],
  ['missing Pack', (value) => { delete value.expectedPack; }],
  ['partial Pack', (value) => { delete value.expectedPack.envelopeDigest; }],
  ['unknown Pack', (value) => { value.expectedPack.schema = 'unknown'; }],
  ['enabled origin', (value) => { value.originPolicy = 'enabled'; }],
  ['missing trust', (value) => { value.openOptions.trustedSigners = {}; }],
  ['malformed trust', (value) => { value.openOptions.trustedSigners = 'signer'; }],
  ['missing plans', (value) => { value.openOptions.acceptedTargetPlanDigests = []; }],
  ['missing assignment', (value) => { delete value.sequenceOptions.assignment; }],
  ['wrong operation options', (value) => { value.sequenceOptions.includeLogits = true; }],
]) {
  const invalid = structuredClone(config);
  mutate(invalid);
  assert.throws(() => validateSequencePackQualificationConfig(invalid), /Qualification/, name);
}

// Retained adverse observations must survive pre-GPU failures; no model is injected.
const originalFetch = globalThis.fetch;
const changedPack = structuredClone(config);
changedPack.expectedPack.envelopeDigest = `sha256:${'0'.repeat(64)}`;
const packFailure = await qualifySequencePack(changedPack);
assert.equal(packFailure.passed, false);
assert.equal(packFailure.stage, 'input-verification');
assert.match(packFailure.error.message, /Pack identity differs/);
assert.deepEqual(JSON.parse(await readFile(config.outputPath, 'utf8')), packFailure);
assert.equal(globalThis.fetch, originalFetch);
const referenceFailure = await qualifySequencePack({ ...config, outputPath: join(directory, 'reference-failure.json') });
assert.equal(referenceFailure.passed, false);
assert.match(referenceFailure.error.message, /Reference digest differs/);
assert.equal(referenceFailure.runtime, undefined);
assert.equal(globalThis.fetch, originalFetch);
console.log('qualify-sequence-pack.test: ok');
