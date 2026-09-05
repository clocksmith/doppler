import assert from 'node:assert/strict';
import { validateSequenceReferenceTranscript } from '../../src/config/sequence-reference.js';
import { buildSequenceReferenceTranscript } from '../../src/tooling/program-bundle/sequence-reference.js';
import { evaluateSequenceReference } from '../../tools/lib/sequence-model-qualification.js';

const digest = (character) => `sha256:${character.repeat(64)}`;
const artifact = { path: 'reference.json', hash: digest('a') };
const graphHash = digest('b');

// Synthetic contract evidence only: this fixture does not qualify a model or GPU.
function createReport() {
  const reference = {
    schema: 'doppler.sequenceModelReference.v1',
    modelId: 'unit-encoder',
    source: { checkpointId: 'unit/encoder@revision', repository: 'unit/encoder', revision: 'revision' },
    input: { sequence: 'M', alphabet: 'protein', tokenIds: [0, 20, 2] },
    outputs: { logits: false },
    probes: {
      pooledEmbedding: { indices: [0, 1], values: [1, 2] },
      tokenEmbeddings: [{ position: 1, indices: [0, 1], values: [3, 4] }],
    },
    tolerances: { pooledEmbeddingMaxAbs: 0.001, tokenEmbeddingMaxAbs: 0.001 },
  };
  const manifest = {
    modelId: reference.modelId,
    artifactIdentity: { sourceCheckpointId: reference.source.checkpointId },
    inference: { supportsSequence: true, sequence: { alphabet: 'protein' } },
  };
  const result = {
    tokens: reference.input.tokenIds,
    embeddingDim: 2,
    pooledEmbedding: new Float32Array([1, 2]),
    tokenEmbeddings: new Float32Array([1, 2, 3, 4, 5, 6]),
    logits: null,
  };
  const evaluation = evaluateSequenceReference({ manifest, result, reference });
  assert.equal(evaluation.passed, true);
  return {
    schema: 'doppler.sequenceModelQualification.v1',
    passed: evaluation.passed,
    model: { modelId: manifest.modelId, manifestHash: digest('c'), artifactIdentity: manifest.artifactIdentity },
    reference: {
      path: 'unit-reference.json', digest: digest('d'), source: reference.source,
      input: reference.input, tolerances: reference.tolerances,
    },
    runtime: { surface: 'node-webgpu', executionGraphHash: graphHash, adapterInfo: { vendor: 'synthetic' } },
    result: {
      options: { includeTokenEmbeddings: true, includeLogits: false },
      embeddingDim: result.embeddingDim, tokenCount: result.tokens.length,
      checks: evaluation.checks, outputDigests: evaluation.outputDigests,
    },
  };
}

const report = createReport();
const built = buildSequenceReferenceTranscript(report, artifact, graphHash);
assert.deepEqual(validateSequenceReferenceTranscript(built.transcript), { ok: true, errors: [] });
assert.equal(built.transcript.operation, 'encodeSequence');
assert.deepEqual(built.transcript.source, { kind: 'sequence-qualification', ...artifact });
assert.deepEqual(built.adapter, { source: 'reference-report', surface: 'node-webgpu', deviceInfo: { vendor: 'synthetic' } });
report.reference.input.tokenIds[0] = 99;
assert.equal(built.transcript.reference.input.tokenIds[0], 0, 'captured evidence must not alias its producer');

function check(transcript, id) {
  return transcript.checks.find((entry) => entry.id === id);
}

const invalidTranscripts = [
  ['wrong operation', (value) => { value.operation = 'generateText'; }],
  ['wrong model', (value) => { value.modelId = 'another-model'; }],
  ['wrong checkpoint', (value) => { value.reference.source.checkpointId = 'another/checkpoint'; }],
  ['wrong actual model', (value) => { check(value, 'model.identity').actualModelId = 'another-model'; }],
  ['wrong expected model', (value) => { check(value, 'model.identity').expectedModelId = 'another-model'; }],
  ['wrong actual checkpoint', (value) => { check(value, 'model.identity').actualCheckpointId = 'another/checkpoint'; }],
  ['wrong expected checkpoint', (value) => { check(value, 'model.identity').expectedCheckpointId = 'another/checkpoint'; }],
  ['wrong input alphabet', (value) => { value.reference.input.alphabet = 'nucleotide'; }],
  ['wrong actual alphabet', (value) => { check(value, 'sequence.contract').actualAlphabet = 'nucleotide'; }],
  ['wrong expected alphabet', (value) => { check(value, 'sequence.contract').expectedAlphabet = 'nucleotide'; }],
  ['reported logits', (value) => { check(value, 'logits.not-requested').actual = 'present'; }],
  ['reported tokenizer mismatch', (value) => { check(value, 'tokenizer.ids').mismatches = [{ index: 0 }]; }],
  ['missing tokenizer mismatches', (value) => { delete check(value, 'tokenizer.ids').mismatches; }],
  ['non-array parity failures', (value) => { check(value, 'pooledEmbedding.parity').failures = ''; }],
  ['wrong output geometry', (value) => { value.output.embeddingDim += 1; }],
  ['wrong token count', (value) => { value.output.tokenCount += 1; }],
  ['non-finite output', (value) => { check(value, 'pooledEmbedding.finite').nonFiniteCount = 1; }],
  ['out-of-tolerance samples', (value) => { check(value, 'tokenEmbeddings.parity').maxAbsoluteError = 0.01; }],
  ['tolerance mismatch', (value) => { value.reference.tolerances.tokenEmbeddingMaxAbs = 0.01; }],
  ['missing samples', (value) => { check(value, 'tokenEmbeddings.parity').sampleCount = 0; }],
  ['failed check', (value) => { check(value, 'model.identity').passed = false; }],
  ['missing check', (value) => { value.checks.pop(); }],
  ['duplicate check', (value) => { value.checks.push(value.checks[0]); }],
  ['null check', (value) => { value.checks[0] = null; }],
  ['missing digest', (value) => { delete value.output.digests.pooledEmbedding; }],
  ['requested logits', (value) => { value.options.includeLogits = true; }],
  ['generation evidence', (value) => { value.generationConfig = {}; }],
];
for (const [name, mutate] of invalidTranscripts) {
  const transcript = structuredClone(built.transcript);
  mutate(transcript);
  assert.equal(validateSequenceReferenceTranscript(transcript).ok, false, name);
}

for (const [name, mutate, expected] of [
  ['failed qualification', (value) => { value.passed = false; }, /passed sequence qualification/],
  ['different graph', (value) => { value.runtime.executionGraphHash = digest('e'); }, /execution graph/],
  ['different checkpoint', (value) => { value.model.artifactIdentity.sourceCheckpointId = 'other'; }, /source identity/],
  ['different model', (value) => { value.model.modelId = 'other'; }, /source identity/],
  ['different alphabet', (value) => { value.reference.input.alphabet = 'nucleotide'; }, /alphabet/],
]) {
  const input = createReport();
  mutate(input);
  assert.throws(() => buildSequenceReferenceTranscript(input, artifact, graphHash), expected, name);
}

console.log('sequence-reference-transcript.test: ok');
