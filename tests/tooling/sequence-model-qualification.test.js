import assert from 'node:assert/strict';

import {
  createSyntheticSequenceLoRAManifest,
  evaluateSequenceLoRAQualification,
  evaluateSequenceReference,
  validateSequenceReference,
} from '../../tools/lib/sequence-model-qualification.js';
import { parseArgs } from '../../tools/qualify-sequence-model.js';

const reference = {
  schema: 'doppler.sequenceModelReference.v1',
  modelId: 'sequence-test',
  source: { checkpointId: 'source@test' },
  input: { alphabet: 'amino_acid', sequence: 'AC', tokenIds: [3, 5, 6, 4] },
  probes: {
    pooledEmbedding: { indices: [0, 1], values: [4, 6] },
    tokenEmbeddings: [
      { position: 1, indices: [0, 1], values: [2, 4] },
      { position: 2, indices: [0, 1], values: [6, 8] },
    ],
    logits: [
      { position: 1, indices: [0, 1], values: [0, 5] },
    ],
    argmaxTokenIds: [0, 1, 2, 0],
  },
  tolerances: {
    pooledEmbeddingMaxAbs: 0.001,
    tokenEmbeddingMaxAbs: 0.001,
    logitMaxAbs: 0.001,
  },
};

validateSequenceReference(reference);

const manifest = {
  modelId: 'sequence-test',
  artifactIdentity: { sourceCheckpointId: 'source@test' },
  inference: { supportsSequence: true, sequence: { alphabet: 'amino_acid' } },
};
const result = {
  tokens: [3, 5, 6, 4],
  embeddingDim: 2,
  vocabSize: 3,
  pooledEmbedding: new Float32Array([4, 6]),
  tokenEmbeddings: new Float32Array([0, 0, 2, 4, 6, 8, 0, 0]),
  logits: new Float32Array([
    3, 2, 1,
    0, 5, 1,
    0, 1, 5,
    3, 2, 1,
  ]),
};

const passing = evaluateSequenceReference({ manifest, result, reference });
assert.equal(passing.passed, true);
assert.equal(passing.checks.every((check) => check.passed), true);
assert.match(passing.outputDigests.logits, /^sha256:[a-f0-9]{64}$/u);

const badResult = {
  ...result,
  logits: new Float32Array(result.logits),
};
badResult.logits[4] = -2;
const failing = evaluateSequenceReference({ manifest, result: badResult, reference });
assert.equal(failing.passed, false);
assert.equal(failing.checks.find((check) => check.id === 'logits.parity').passed, false);
assert.equal(failing.checks.find((check) => check.id === 'logits.argmax').passed, false);

const embeddingOnlyReference = {
  ...reference,
  outputs: { logits: false },
  probes: {
    pooledEmbedding: reference.probes.pooledEmbedding,
    tokenEmbeddings: reference.probes.tokenEmbeddings,
  },
  tolerances: {
    pooledEmbeddingMaxAbs: reference.tolerances.pooledEmbeddingMaxAbs,
    tokenEmbeddingMaxAbs: reference.tolerances.tokenEmbeddingMaxAbs,
  },
};
const embeddingOnly = evaluateSequenceReference({
  manifest,
  result: { ...result, logits: null },
  reference: embeddingOnlyReference,
});
assert.equal(embeddingOnly.passed, true);
assert.equal(embeddingOnly.outputDigests.logits, null);
assert.equal(
  embeddingOnly.checks.find((check) => check.id === 'logits.not-requested').passed,
  true
);

assert.throws(
  () => validateSequenceReference({ ...reference, schema: 'unknown' }),
  /Unsupported sequence reference schema/
);

const parsed = parseArgs([
  '--model-dir', '/tmp/model',
  '--qualify-lora',
  '--diagnose-layer', '0',
  '--diagnose-layer', '3',
  '--diagnose-op', 'embed.out',
]);
assert.deepEqual(parsed.diagnoseLayers, [0, 3]);
assert.deepEqual(parsed.diagnoseOps, ['embed.out']);
assert.equal(parsed.qualifyLoRA, true);
assert.match(parsed.reference, /amplify-120m-sequence-reference\.json$/u);
assert.throws(
  () => parseArgs(['--model-dir', '/tmp/model', '--diagnose-layer', '-1']),
  /non-negative integer/
);

const syntheticManifest = createSyntheticSequenceLoRAManifest({
  modelId: 'sequence-test',
  architecture: {
    hiddenSize: 2,
    numAttentionHeads: 1,
    headDim: 2,
  },
});
assert.equal(syntheticManifest.baseModel, 'sequence-test');
assert.deepEqual(syntheticManifest.tensors.map((tensor) => tensor.shape), [[1, 2], [2, 1]]);

const adaptedResult = {
  ...result,
  pooledEmbedding: new Float32Array([4.1, 6]),
  tokenEmbeddings: new Float32Array([0, 0, 2.1, 4, 6, 8, 0, 0]),
  logits: new Float32Array(result.logits),
};
adaptedResult.logits[0] += 0.1;
const loraQualification = evaluateSequenceLoRAQualification({
  baseResult: result,
  adaptedResult,
  restoredResult: result,
  expectedAdapterName: syntheticManifest.name,
  activeAdapterName: syntheticManifest.name,
  unloadedAdapterName: null,
  wrongBaseError: 'LoRA adapter targets base model "wrong", but the loaded model is "sequence-test".',
  invalidLayerError: 'LoRA adapter targets layer 12, which is absent from the loaded model.',
});
assert.equal(loraQualification.passed, true);
assert.equal(
  loraQualification.checks.find((check) => check.id === 'lora.changed.logits').passed,
  true
);

console.log('sequence-model-qualification.test: ok');
