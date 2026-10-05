import assert from 'node:assert/strict';
import { assertChoiceScoringReferenceTranscript } from '../../src/config/choice-scoring-reference.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { buildChoiceScoringReferenceTranscript } from '../../src/tooling/program-bundle/choice-scoring-reference.js';
import { buildQualificationRecords } from '../../src/converter/forge-qualification.js';

const digest = `sha256:${'1'.repeat(64)}`;
const row = { id: 'synthetic', input: { prompt: 'Choose:', choices: [{ id: 'yes', label: ' A' }, { id: 'no', label: ' B' }], maxSeqLen: 8 },
  output: { schema: 'doppler.choice-scores/v1', interpretation: 'next-token-logits', calibration: null,
    choices: [{ id: 'yes', label: ' A', tokenId: 3, logit: 2 }, { id: 'no', label: ' B', tokenId: 4, logit: 1 }],
    selectedId: 'yes', promptTokenCount: 2 }, expectedId: 'yes', promptTokenIds: [1, 2] };
const reference = { schema: 'doppler.choice-scoring-source-reference/v1', engine: 'synthetic independent operands',
  engineVersions: { fixture: '1' }, contractHash: digest, manifestHash: digest,
  maximumAbsoluteLogitError: 0.05, minimumCorrectChoices: 1, cases: [row] };
const report = { schema: 'doppler.choiceScoringModelQualification.v1', passed: true,
  model: { modelId: 'fixture', manifestHash: digest },
  runtime: { executionGraphHash: digest, surface: 'test-webgpu', adapterInfo: { synthetic: true } },
  reference, referenceDigest: computeCanonicalSha256(reference), observation: { cases: structuredClone(reference.cases) } };
const artifact = { path: 'fixture.json', hash: digest };
const transcript = buildChoiceScoringReferenceTranscript(report, artifact, digest).transcript;
assert.equal(assertChoiceScoringReferenceTranscript(transcript), transcript);
for (const change of [
  value => { value.referenceDigest = `sha256:${'2'.repeat(64)}`; },
  value => { value.manifestHash = `sha256:${'2'.repeat(64)}`; },
  value => { value.observation.cases = []; },
  value => { value.observation.cases[0].input.prompt = 'different'; },
  value => { value.observation.cases[0].promptTokenIds[0] = 7; },
  value => { value.observation.cases[0].output.promptTokenCount = 3; },
  value => { value.observation.cases[0].output.choices[0].tokenId = 7; },
  value => { value.observation.cases[0].output.choices[0].logit += 0.1; },
  value => { value.observation.cases[0].output.selectedId = 'no'; },
  value => { value.observation.cases[0].expectedId = 'no'; },
  value => { value.tokens = { ids: [1] }; },
]) {
  const changed = structuredClone(transcript); change(changed);
  assert.throws(() => assertChoiceScoringReferenceTranscript(changed));
}
const wrongTask = structuredClone(transcript);
wrongTask.reference.cases[0].expectedId = 'no';
wrongTask.observation.cases[0].expectedId = 'no';
wrongTask.referenceDigest = computeCanonicalSha256(wrongTask.reference);
assert.throws(() => assertChoiceScoringReferenceTranscript(wrongTask), /task quality/);
assert.throws(() => buildChoiceScoringReferenceTranscript({ ...report, passed: false }, artifact, digest), /passed/);
assert.throws(() => buildChoiceScoringReferenceTranscript(report, artifact, `sha256:${'2'.repeat(64)}`), /exact execution graph/);
const lowered = { modelIR: { modelId: 'fixture', outputTopology: { headType: 'causal-lm' } },
  normalized: { artifacts: [{ ...artifact, artifactId: 'fixture', role: 'reference-report' }], manifestHash: digest,
    qualificationEvidence: [], programBundle: { referenceTranscript: transcript,
      captureProfile: { surfaces: ['test-webgpu'] }, execution: { graphHash: digest } } } };
const records = buildQualificationRecords(lowered);
assert.equal(records[0].operation, 'scoreChoices');
assert.equal(records[0].scoredChoices, 2);
assert.equal(records[0].generatedTokens, undefined);
const encoder = structuredClone(lowered); encoder.modelIR.outputTopology.headType = 'sequence-encoder';
assert.throws(() => buildQualificationRecords(encoder), /causal LM/);
const extraSurface = structuredClone(lowered); extraSurface.normalized.programBundle.captureProfile.surfaces.push('node-webgpu');
assert.throws(() => buildQualificationRecords(extraSurface), /actual qualification/);
console.log('choice-scoring-reference.test.js passed (synthetic qualification validation)');
