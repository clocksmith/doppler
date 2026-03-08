import assert from 'node:assert/strict';

import {
  computeManifestHash,
  verifyIntentBundle,
} from '../../src/hotswap/intent-bundle.js';

const manifest = {
  modelId: 'intent-test',
  modelType: 'transformer',
  quantization: 'Q4_K_M',
  hashAlgorithm: 'sha256',
  totalSize: 1,
  architecture: {
    numLayers: 1,
    hiddenSize: 64,
    intermediateSize: 256,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 64,
    vocabSize: 32000,
    maxSeqLen: 1024,
  },
  inference: {
    presetId: 'gemma3',
  },
  shards: [],
};

const manifestHash = await computeManifestHash(manifest);
const bundle = {
  foundation: {
    baseModelHash: `sha256:${manifestHash}`,
    kernelRegistryVersion: '2026.03.07',
  },
  constraints: {},
  payload: {},
};

{
  const verification = await verifyIntentBundle(bundle, {});
  assert.equal(verification.ok, false);
  assert.match(
    verification.reasons.join('; '),
    /Missing verification context manifest/
  );
  assert.match(
    verification.reasons.join('; '),
    /Missing verification context kernelRegistryVersion/
  );
}

{
  const verification = await verifyIntentBundle(bundle, {
    manifest,
    kernelRegistryVersion: '2026.03.07',
  });
  assert.equal(verification.ok, true);
}

console.log('hotswap-intent-bundle-contract.test: ok');
