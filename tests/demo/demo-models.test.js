import assert from 'node:assert/strict';

import {
  DEFAULT_EXECUTION_V1_SESSION,
  EXECUTION_V1_SCHEMA_ID,
} from '../../src/config/schema/index.js';
import {
  assertWeightsRefPrimaryAvailable,
  buildModelCardDetail,
  buildLocalModelBaseUrl,
  buildModelSourceCandidates,
  buildRemoveConfirmText,
  canRemoveModelStatus,
  findPrimaryForWeightPack,
  findRegisteredSiblingsOf,
  patchManifestCompat,
  selectDemoCatalogEntries,
} from '../../demo/models.js';

const DEMO_READY = Object.freeze({
  artifactCompleteness: 'complete',
  runtimePromotionState: 'manifest-owned',
  weightsRefAllowed: false,
});

const DEMO_WEIGHTS_REF_READY = Object.freeze({
  artifactCompleteness: 'weights-ref',
  runtimePromotionState: 'manifest-owned',
  weightsRefAllowed: true,
});

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
      ...DEMO_READY,
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
      ...DEMO_READY,
      modelId: 'hidden-hosted-text-model',
      quickstart: true,
      demoVisible: false,
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/hidden-hosted-text-model',
      },
      sortOrder: 13,
    },
    {
      ...DEMO_READY,
      modelId: 'qwen-3-6-27b-q4k-ehaf16',
      quickstart: false,
      demoVisible: true,
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/qwen-3-6-27b-q4k-ehaf16',
      },
      sortOrder: 22,
    },
    {
      ...DEMO_READY,
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      quickstart: false,
      modes: ['translate'],
      sortOrder: 14,
    },
  ], {
    localBaseUrls: new Map([
      ['qwen-3-6-27b-q4k-ehaf16', 'http://localhost:8080/models/local/qwen-3-6-27b-q4k-ehaf16'],
    ]),
  });

  assert.deepEqual(
    selected.map((entry) => entry.modelId),
    ['gemma-3-1b-it-q4k-ehf16-af32', 'qwen-3-6-27b-q4k-ehaf16'],
    'demo catalog should include quickstart text models and explicitly demo-visible text models'
  );
  assert.equal(
    selected[1].localBaseUrl,
    'http://localhost:8080/models/local/qwen-3-6-27b-q4k-ehaf16'
  );
}

{
  const selected = selectDemoCatalogEntries([
    {
      ...DEMO_READY,
      modelId: 'missing-source-text-model',
      quickstart: true,
      modes: ['text'],
      sortOrder: 20,
    },
    {
      ...DEMO_READY,
      modelId: 'hf-backed-text-model',
      quickstart: true,
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
  const selected = selectDemoCatalogEntries([
    {
      ...DEMO_READY,
      modelId: 'gemma-4-31b-it-text-q4k-ehf16-af32',
      weightPackId: 'gemma-4-31b-it-text-q4k-ehf16-af32-wp-catalog-v1',
      quickstart: false,
      demoVisible: true,
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/gemma-4-31b-it-text-q4k-ehf16-af32',
      },
      sortOrder: 16,
    },
    {
      ...DEMO_WEIGHTS_REF_READY,
      modelId: 'gemma-4-31b-it-text-q4k-ehf16-af16',
      weightPackId: 'gemma-4-31b-it-text-q4k-ehf16-af32-wp-catalog-v1',
      quickstart: false,
      demoVisible: true,
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/gemma-4-31b-it-text-q4k-ehf16-af16',
      },
      sortOrder: 17,
    },
  ]);

  assert.deepEqual(
    selected.map((entry) => entry.modelId),
    [
      'gemma-4-31b-it-text-q4k-ehf16-af32',
      'gemma-4-31b-it-text-q4k-ehf16-af16',
    ],
    'demo catalog should include manifest-only siblings when their primary weight pack is available'
  );
  assert.equal(
    selected[1].weightsRefPrimary,
    'gemma-4-31b-it-text-q4k-ehf16-af32'
  );
}

{
  const selected = selectDemoCatalogEntries([
    {
      ...DEMO_WEIGHTS_REF_READY,
      modelId: 'orphan-af16',
      weightPackId: 'missing-weight-pack',
      quickstart: false,
      demoVisible: true,
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/orphan-af16',
      },
      sortOrder: 18,
    },
  ]);

  assert.deepEqual(
    selected.map((entry) => entry.modelId),
    [],
    'demo catalog should exclude manifest-only siblings without a reachable primary weight pack'
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
  assert.equal(
    buildRemoveConfirmText({ artifactCompleteness: 'weights-ref' }),
    'Remove this manifest from OPFS? Shared weights remain.'
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
  assert.equal(
    buildModelCardDetail({
      artifactCompleteness: 'weights-ref',
      weightsRefPrimary: 'gemma-4-31b-it-text-q4k-ehf16-af32',
    }, 'stored'),
    'Ready · Shared with gemma-4-31b-it-text-q4k-ehf16-af32'
  );
}

// findPrimaryForWeightPack: locates the primary lane that owns a weight pack.
{
  const catalog = [
    { modelId: 'primary', weightPackId: 'wp-1', artifactCompleteness: 'complete', weightsRefAllowed: false },
    { modelId: 'sibling', weightPackId: 'wp-1', artifactCompleteness: 'weights-ref', weightsRefAllowed: true },
    { modelId: 'other-primary', weightPackId: 'wp-2', artifactCompleteness: 'complete', weightsRefAllowed: false },
  ];
  assert.equal(findPrimaryForWeightPack(catalog, 'wp-1')?.modelId, 'primary');
  assert.equal(findPrimaryForWeightPack(catalog, 'wp-2')?.modelId, 'other-primary');
  assert.equal(findPrimaryForWeightPack(catalog, 'wp-missing'), null);
  assert.equal(findPrimaryForWeightPack(null, 'wp-1'), null);
  assert.equal(findPrimaryForWeightPack(catalog, ''), null);
}

// assertWeightsRefPrimaryAvailable: passes for primary lanes; passes for
// siblings only when the primary is in OPFS; throws with primary modelId
// in the error message otherwise.
{
  const catalog = [
    { modelId: 'primary', weightPackId: 'wp-1', artifactCompleteness: 'complete', weightsRefAllowed: false },
    { modelId: 'sibling', weightPackId: 'wp-1', artifactCompleteness: 'weights-ref', weightsRefAllowed: true },
    { modelId: 'orphan', weightPackId: 'wp-missing', artifactCompleteness: 'weights-ref', weightsRefAllowed: true },
  ];

  // Primary lane: nothing to assert.
  assert.equal(
    assertWeightsRefPrimaryAvailable(catalog[0], catalog, new Set()),
    null,
    'primary lane is a no-op',
  );

  // Sibling with primary stored: returns the primary entry.
  const result = assertWeightsRefPrimaryAvailable(catalog[1], catalog, new Set(['primary']));
  assert.equal(result?.modelId, 'primary');

  // Sibling without primary stored: throws with primary id named.
  assert.throws(
    () => assertWeightsRefPrimaryAvailable(catalog[1], catalog, new Set()),
    /shares weights with primary\. Download primary first/,
  );

  // Orphan sibling (no primary in catalog): throws with weightPackId named.
  assert.throws(
    () => assertWeightsRefPrimaryAvailable(catalog[2], catalog, new Set()),
    /no primary lane in the catalog \(weightPackId=wp-missing\)/,
  );
}

// findRegisteredSiblingsOf: enumerates registered weights-ref siblings that
// depend on a primary. Used by the remove-stored-model guard.
{
  const catalog = [
    { modelId: 'primary', weightPackId: 'wp-1', artifactCompleteness: 'complete', weightsRefAllowed: false },
    { modelId: 'sibling-a', weightPackId: 'wp-1', artifactCompleteness: 'weights-ref', weightsRefAllowed: true },
    { modelId: 'sibling-b', weightPackId: 'wp-1', artifactCompleteness: 'weights-ref', weightsRefAllowed: true },
    { modelId: 'unrelated', weightPackId: 'wp-2', artifactCompleteness: 'weights-ref', weightsRefAllowed: true },
  ];
  // No siblings registered: empty.
  assert.deepEqual(
    findRegisteredSiblingsOf(catalog[0], catalog, new Set()).map((e) => e.modelId),
    [],
  );
  // One sibling registered: just that one.
  assert.deepEqual(
    findRegisteredSiblingsOf(catalog[0], catalog, new Set(['sibling-a'])).map((e) => e.modelId),
    ['sibling-a'],
  );
  // Both siblings registered: both reported.
  assert.deepEqual(
    findRegisteredSiblingsOf(catalog[0], catalog, new Set(['sibling-a', 'sibling-b'])).map((e) => e.modelId),
    ['sibling-a', 'sibling-b'],
  );
  // Unrelated sibling on a different weight pack: ignored.
  assert.deepEqual(
    findRegisteredSiblingsOf(catalog[0], catalog, new Set(['unrelated'])).map((e) => e.modelId),
    [],
  );
  // Calling with a non-primary entry returns empty.
  assert.deepEqual(
    findRegisteredSiblingsOf(catalog[1], catalog, new Set(['sibling-a', 'sibling-b'])).map((e) => e.modelId),
    [],
  );
}

console.log('demo-models.test: ok');
