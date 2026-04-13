import assert from 'node:assert/strict';

import {
  DEFAULT_EXECUTION_V1_SESSION,
  EXECUTION_V1_SCHEMA_ID,
} from '../../src/config/schema/index.js';
import {
  buildModelCardDetail,
  buildLocalModelBaseUrl,
  buildModelSourceCandidates,
  buildRemoveConfirmText,
  canRemoveModelStatus,
  patchManifestCompat,
  selectDemoCatalogEntries,
} from '../../demo/models.js';

{
  const patched = patchManifestCompat({
    modelId: 'legacy-gemma',
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      attention: {},
      normalization: {},
      ffn: {},
      rope: {},
      output: {},
      layerPattern: {},
      chatTemplate: {},
      session: {
        compute: {
          defaults: {
            activationDtype: 'f32',
            mathDtype: 'f32',
            accumDtype: 'f32',
            outputDtype: 'f32',
          },
        },
        kvcache: null,
        decodeLoop: null,
      },
    },
  });

  assert.deepEqual(
    patched.inference.session.perLayerInputs,
    DEFAULT_EXECUTION_V1_SESSION.perLayerInputs,
    'execution-v1 compat patch must backfill missing per-layer input session policy'
  );
}

{
  const patched = patchManifestCompat({
    modelId: 'legacy-gemma',
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      attention: {},
      normalization: {},
      ffn: {},
      rope: {},
      output: {},
      layerPattern: {},
      chatTemplate: {},
      session: {
        compute: {
          defaults: {
            activationDtype: 'f32',
            mathDtype: 'f32',
            accumDtype: 'f32',
            outputDtype: 'f32',
          },
        },
        kvcache: null,
        decodeLoop: null,
        perLayerInputs: {
          ...DEFAULT_EXECUTION_V1_SESSION.perLayerInputs,
          materialization: 'range_backed',
        },
      },
    },
  });

  assert.equal(
    patched.inference.session.perLayerInputs.materialization,
    'range_backed',
    'compat patch must preserve explicit per-layer input session policy'
  );
}

{
  const selected = selectDemoCatalogEntries([
    {
      modelId: 'gemma-3-1b-it-q4k-ehf16-af32',
      quickstart: true,
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/gemma-3-1b-it-q4k-ehf16-af32',
      },
      sortOrder: 12,
    },
    {
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      quickstart: false,
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/gemma-4-e2b-it-q4k-ehf16-af32',
      },
      sortOrder: 13,
    },
    {
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      quickstart: false,
      modes: ['translate'],
      sortOrder: 14,
    },
  ], {
    localBaseUrls: new Map([
      ['gemma-4-e2b-it-q4k-ehf16-af32', 'http://localhost:8080/models/local/gemma-4-e2b-it-q4k-ehf16-af32'],
    ]),
  });

  assert.deepEqual(
    selected.map((entry) => entry.modelId),
    ['gemma-3-1b-it-q4k-ehf16-af32', 'gemma-4-e2b-it-q4k-ehf16-af32'],
    'demo catalog should include downloadable text models regardless of quickstart metadata'
  );
  assert.equal(
    selected[1].localBaseUrl,
    'http://localhost:8080/models/local/gemma-4-e2b-it-q4k-ehf16-af32'
  );
}

{
  const selected = selectDemoCatalogEntries([
    {
      modelId: 'missing-source-text-model',
      modes: ['text'],
      sortOrder: 20,
    },
    {
      modelId: 'hf-backed-text-model',
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/hf-backed-text-model',
      },
      sortOrder: 21,
    },
  ]);

  assert.deepEqual(
    selected.map((entry) => entry.modelId),
    ['hf-backed-text-model'],
    'demo catalog should exclude text entries that do not resolve to a downloadable source'
  );
}

{
  const origin = 'http://localhost:8080';
  const localBaseUrl = buildLocalModelBaseUrl('gemma-4-e2b-it-q4k-ehf16-af32', origin);
  assert.equal(
    localBaseUrl,
    'http://localhost:8080/models/local/gemma-4-e2b-it-q4k-ehf16-af32'
  );

  const candidates = buildModelSourceCandidates({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    localBaseUrl,
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: 'abc123',
      path: 'models/gemma-4-e2b-it-q4k-ehf16-af32',
    },
  });
  assert.deepEqual(
    candidates.map((candidate) => candidate.kind),
    ['local', 'hf'],
    'demo download should prefer same-origin local artifacts before hosted HF'
  );
}

{
  assert.equal(canRemoveModelStatus('stored'), true);
  assert.equal(canRemoveModelStatus('loaded'), true);
  assert.equal(canRemoveModelStatus('available'), false);
  assert.equal(canRemoveModelStatus('downloading'), false);
}

{
  assert.equal(
    buildRemoveConfirmText({ sizeBytes: 612 * 1024 * 1024 }),
    'Remove 612 MB from OPFS?'
  );
  assert.equal(
    buildRemoveConfirmText({ sizeBytes: 0 }),
    'Remove this model from OPFS?'
  );
}

{
  assert.equal(
    buildModelCardDetail({ sizeBytes: 612 * 1024 * 1024 }, 'stored'),
    'Ready · 612 MB'
  );
  assert.equal(
    buildModelCardDetail({ sizeBytes: 1536 * 1024 * 1024 }, 'loaded'),
    'Active · 1.5 GB'
  );
  assert.equal(
    buildModelCardDetail({ sizeBytes: 612 * 1024 * 1024 }, 'downloading'),
    'Downloading...'
  );
}

console.log('demo-models.test: ok');
