import assert from 'node:assert/strict';

import {
  buildCurrentInferenceStatusBuckets,
  resolveRowStatus,
  validateCatalogMatrixInputs,
} from '../../tools/sync-model-support-matrix.js';

{
  assert.deepEqual(validateCatalogMatrixInputs({
    updatedAt: '2026-03-06',
    models: [
      {
        modelId: 'gemma-3-270m-it-wq4k-ef16-hf16',
        baseUrl: './curated/gemma-3-270m-it-wq4k-ef16-hf16',
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: '4efe64a914892e98be50842aeb16c3b648cc68a5',
          path: 'models/gemma-3-270m-it-wq4k-ef16',
        },
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            demo: 'curated',
          },
        },
      },
    ],
  }), []);
}

{
  assert.deepEqual(validateCatalogMatrixInputs({
    updatedAt: '',
    models: [
      {
        modelId: 'broken-model',
        baseUrl: null,
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: '',
          path: '',
        },
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            demo: 'curated',
          },
        },
      },
      {
        modelId: 'broken-model',
        baseUrl: null,
        lifecycle: {
          status: {
            demo: 'local',
          },
        },
      },
    ],
  }), [
    'catalog updatedAt must be a non-empty string',
    'broken-model: lifecycle.availability.hf=true requires hf.revision',
    'broken-model: lifecycle.availability.hf=true requires hf.path',
    'broken-model: lifecycle.status.demo=curated requires a curated baseUrl',
    'duplicate catalog modelId: broken-model',
    'broken-model: lifecycle.status.demo=local requires a local baseUrl',
  ]);
}

{
  assert.equal(resolveRowStatus({
    conversionCount: 1,
    runtimeStatus: 'active',
    catalogCount: 1,
    lifecycleTested: 'unknown',
  }), 'verification-pending');

  assert.equal(resolveRowStatus({
    conversionCount: 1,
    runtimeStatus: 'active',
    catalogCount: 1,
    lifecycleTested: 'verified',
  }), 'verified');

  assert.equal(resolveRowStatus({
    conversionCount: 1,
    runtimeStatus: 'active',
    catalogCount: 1,
    lifecycleTested: 'failed',
  }), 'verification-failed');
}

{
  const buckets = buildCurrentInferenceStatusBuckets({
    catalogModels: [
      {
        modelId: 'verified-model',
        preset: 'gemma3',
        modes: ['run'],
        sortOrder: 1,
        lifecycle: {
          status: {
            runtime: 'active',
            tested: 'verified',
          },
          tested: {
            result: 'pass',
            lastVerifiedAt: '2026-03-06',
            surface: 'auto',
          },
        },
      },
      {
        modelId: 'unknown-model',
        preset: 'gemma3',
        modes: ['run'],
        sortOrder: 2,
        lifecycle: {
          status: {
            runtime: 'active',
            tested: 'unknown',
          },
        },
      },
      {
        modelId: 'failing-model',
        preset: 'qwen3',
        modes: ['run'],
        sortOrder: 3,
        lifecycle: {
          status: {
            runtime: 'active',
            tested: 'failing',
          },
          tested: {
            result: 'fail',
            lastVerifiedAt: '2026-03-06',
            notes: 'Loads but produces incoherent output.',
          },
        },
      },
    ],
    quickStartModelIds: ['quickstart-only-model', 'verified-model'],
    rows: [
      {
        presetId: 'mamba',
        catalogCount: 0,
        runtimeStatus: 'blocked',
        status: 'blocked-runtime',
      },
      {
        presetId: 'functiongemma',
        catalogCount: 0,
        runtimeStatus: 'active',
        status: 'conversion-ready',
      },
    ],
  });

  assert.equal(buckets.verified.length, 1);
  assert.equal(buckets.verified[0].modelId, 'verified-model');
  assert.equal(buckets.loadsButUnverified.length, 1);
  assert.equal(buckets.loadsButUnverified[0].modelId, 'unknown-model');
  assert.equal(buckets.knownFailing.length, 1);
  assert.equal(buckets.knownFailing[0].modelId, 'failing-model');
  assert.equal(buckets.quickstartOnly.length, 1);
  assert.equal(buckets.quickstartOnly[0].modelId, 'quickstart-only-model');
  assert.equal(buckets.everythingElse.length, 2);
}

console.log('support-matrix-contract.test: ok');
