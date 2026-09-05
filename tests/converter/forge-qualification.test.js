import assert from 'node:assert/strict';
import { buildQualificationRecords } from '../../src/converter/forge-qualification.js';
import { sha256Hex } from '../../src/formats/sha256.js';
import { stableSortObject } from '../../src/formats/stable-sort-object.js';

const digest = `sha256:${'1'.repeat(64)}`;
const hash = (value) => `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
const referenceArtifact = { role: 'reference-report', artifactId: 'reference-fixture', hash: digest };
const generation = {
  modelIR: { modelId: 'fixture', outputTopology: { headType: 'causal-lm' } },
  normalized: {
    artifacts: [referenceArtifact], manifestHash: digest, qualificationEvidence: [],
    programBundle: {
      execution: { graphHash: digest }, captureProfile: { surfaces: ['test-webgpu'] },
      referenceTranscript: { generationConfig: { temperature: 0 }, tokens: { ids: [1, 2] } },
    },
  },
};
const generationRecord = buildQualificationRecords(generation)[0];
assert.deepEqual(generationRecord, {
  surface: 'test-webgpu', status: 'passed', evidenceArtifactId: 'reference-fixture',
  evidenceHash: digest, generatedTokens: 2,
  transcriptHash: hash({ surface: 'test-webgpu', captureProfile: generation.normalized.programBundle.captureProfile,
    transcript: generation.normalized.programBundle.referenceTranscript }),
});
const mutate = (fixture, change) => { const copy = structuredClone(fixture); change(copy); return copy; };
assert.throws(() => buildQualificationRecords(mutate(generation, (v) => { v.normalized.artifacts = []; })), /reference-report/);
assert.throws(() => buildQualificationRecords(mutate(generation, (v) => {
  v.normalized.programBundle.referenceTranscript.generationConfig.temperature = 1;
})), /without a seed/);

// Synthetic encoder contract: this fixture proves validation, not hardware parity.
const sequence = mutate(generation, (v) => {
  v.modelIR.outputTopology.headType = 'sequence-encoder';
  v.normalized.programBundle.referenceTranscript = {
    schema: 'doppler.sequence-reference-transcript/v1', operation: 'encodeSequence',
    modelId: 'fixture', surface: 'test-webgpu', executionGraphHash: digest, manifestHash: digest,
    source: { path: 'fixture.json', hash: digest },
    reference: {
      digest, source: { checkpointId: 'fixture', repository: 'fixture/model', revision: 'fixture-revision' },
      input: { sequence: 'M', alphabet: 'amino_acid', tokenIds: [1] },
      tolerances: { pooledEmbeddingMaxAbs: 0, tokenEmbeddingMaxAbs: 0 },
    },
    options: { includeTokenEmbeddings: true, includeLogits: false },
    output: { embeddingDim: 1, tokenCount: 1, digests: { pooledEmbedding: digest, tokenEmbeddings: digest, logits: null } },
    checks: [
      { id: 'model.identity', passed: true, actualModelId: 'fixture', expectedModelId: 'fixture',
        actualCheckpointId: 'fixture', expectedCheckpointId: 'fixture' },
      { id: 'sequence.contract', passed: true, actualAlphabet: 'amino_acid', expectedAlphabet: 'amino_acid' },
      { id: 'tokenizer.ids', passed: true, expectedCount: 1, actualCount: 1, mismatchCount: 0, mismatches: [] },
      { id: 'pooledEmbedding.finite', passed: true, valueCount: 1, nonFiniteCount: 0 },
      { id: 'tokenEmbeddings.finite', passed: true, valueCount: 1, nonFiniteCount: 0 },
      { id: 'logits.not-requested', passed: true, actual: null },
      { id: 'pooledEmbedding.parity', passed: true, tolerance: 0, sampleCount: 1, maxAbsoluteError: 0, failures: [] },
      { id: 'tokenEmbeddings.parity', passed: true, tolerance: 0, sampleCount: 1, maxAbsoluteError: 0, failures: [] },
    ],
  };
});
assert.deepEqual(buildQualificationRecords(sequence), [{
  surface: 'test-webgpu', status: 'passed', operation: 'encodeSequence', encodedSequences: 1,
  evidenceArtifactId: 'reference-fixture', evidenceHash: digest,
  transcriptHash: hash(sequence.normalized.programBundle.referenceTranscript),
}]);
for (const field of ['modelId', 'manifestHash', 'executionGraphHash']) {
  assert.throws(() => buildQualificationRecords(mutate(sequence, (v) => {
    v.normalized.programBundle.referenceTranscript[field] = field === 'modelId' ? 'another-model' : `sha256:${'2'.repeat(64)}`;
    if (field === 'modelId') {
      const identity = v.normalized.programBundle.referenceTranscript.checks.find((check) => check.id === 'model.identity');
      identity.actualModelId = 'another-model';
      identity.expectedModelId = 'another-model';
    }
  })), /does not match/);
}
assert.throws(() => buildQualificationRecords(mutate(sequence, (v) => {
  v.normalized.programBundle.captureProfile.surfaces.push('another-webgpu');
})), /capture surface/);
assert.throws(() => buildQualificationRecords(mutate(sequence, (v) => {
  v.normalized.programBundle.referenceTranscript.checks[0].passed = false;
})), /Invalid sequence reference/);
assert.throws(() => buildQualificationRecords(mutate(generation, (v) => {
  v.modelIR.outputTopology.headType = 'sequence-encoder';
})), /generation evidence is insufficient/);
console.log('forge-qualification.test: ok (synthetic generation and sequence records)');
