import assert from 'node:assert/strict';

import { validateCatalogMatrixInputs } from '../../tools/sync-model-support-matrix.js';

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

console.log('support-matrix-contract.test: ok');
