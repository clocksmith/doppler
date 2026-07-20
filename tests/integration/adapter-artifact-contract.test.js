import assert from 'node:assert/strict';

import {
  adapterArtifactCacheKey,
  buildImmutableArtifactUrl,
  validateAdapterArtifactOrigin,
  validateAdapterArtifactRecord,
} from '../../src/experimental/adapters/artifact-contract.js';

const hash = (character) => `sha256:${character.repeat(64)}`;

const record = {
  schema: 'doppler.adapter-artifact/v1',
  artifactId: 'unit-adapter',
  lifecycle: 'qualified',
  access: 'private',
  weights: { sha256: hash('a'), bytes: 128, format: 'safetensors' },
  adapterManifest: {
    id: 'unit-adapter',
    name: 'Unit adapter',
    baseModel: 'unit-runtime-model',
    rank: 8,
    alpha: 16,
    targetModules: ['q_proj'],
    checksum: 'a'.repeat(64),
    checksumAlgorithm: 'sha256',
    weightsFormat: 'safetensors',
    weightsPath: 'adapter.safetensors',
    weightsSize: 128,
  },
  trainingBase: {
    repoId: 'clocksmith/unit-source',
    revision: '1'.repeat(40),
  },
  runtimeBase: {
    modelId: 'unit-runtime-model',
    modelSha256: hash('b'),
    manifestSha256: hash('c'),
    tokenizerSha256: hash('d'),
    weightPackId: 'unit-weight-pack',
    weightPackSha256: hash('e'),
    manifestVariantId: 'unit-manifest-variant',
    conversionConfigSha256: hash('f'),
  },
  primaryOrigin: {
    provider: 'gcs',
    bucket: 'clocksmith-adapters-private',
    object: 'v1/adapters/unit-runtime-model/unit-adapter/adapter.safetensors',
    generation: '1720000000000000',
  },
  preservationMirrors: [],
  evidence: [{ kind: 'parity', path: 'reports/unit-parity.json', sha256: hash('9') }],
};

assert.equal(validateAdapterArtifactRecord(record).valid, true);
assert.equal(adapterArtifactCacheKey(record), hash('a'));
assert.equal(
  buildImmutableArtifactUrl(record.primaryOrigin),
  'https://storage.googleapis.com/clocksmith-adapters-private/v1/adapters/unit-runtime-model/unit-adapter/adapter.safetensors?generation=1720000000000000'
);

const huggingFaceOrigin = {
  provider: 'huggingface',
  repoId: 'clocksmith/lora-unit',
  revision: '2'.repeat(40),
  path: 'adapters/unit/adapter_model.safetensors',
};
assert.equal(validateAdapterArtifactOrigin(huggingFaceOrigin).valid, true);
assert.equal(
  buildImmutableArtifactUrl(huggingFaceOrigin),
  `https://huggingface.co/clocksmith/lora-unit/resolve/${'2'.repeat(40)}/adapters/unit/adapter_model.safetensors`
);

assert.equal(validateAdapterArtifactRecord({ ...record, primaryOrigin: null }).valid, false);
assert.equal(validateAdapterArtifactRecord({
  ...record,
  adapterManifest: { ...record.adapterManifest, baseModel: 'wrong-model' },
}).valid, false);
assert.equal(validateAdapterArtifactOrigin({
  provider: 'gcs',
  bucket: 'clocksmith-adapters-private',
  object: '../escape',
  generation: 'latest',
}).valid, false);
assert.equal(validateAdapterArtifactOrigin({
  ...record.primaryOrigin,
  object: 'adapter.bin?generation=latest',
}).valid, false);

console.log('adapter artifact contract tests passed');
