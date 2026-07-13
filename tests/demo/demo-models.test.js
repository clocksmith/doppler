import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

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
  selectDefaultStoredModel,
  selectDemoExecutionEntryForCapabilities,
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
  lifecycle: {
    status: {
      runtime: 'active',
      tested: 'verified',
    },
  },
});

function readManifest(modelId) {
  return JSON.parse(readFileSync(`models/local/${modelId}/manifest.json`, 'utf8'));
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
      demoVisible: true,
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
    'demo catalog should include quickstart models and explicitly demo-visible models'
  );
  assert.equal(
    selected[1].localBaseUrl,
    'http://localhost:8080/models/local/qwen-3-6-27b-q4k-ehaf16'
  );
}

{
  const catalog = JSON.parse(readFileSync('models/catalog.json', 'utf8'));
  const selected = selectDemoCatalogEntries(catalog.models);
  assert.deepEqual(
    selected.map((entry) => entry.modelId),
    [
      'gemma-3-270m-it-q4k-ehf16-af32',
      'gemma-3-1b-it-q4k-ehf16-af32',
      'gemma-4-e2b-it-q4k-ehf16-af16-int4ple',
      'translategemma-4b-1b-enes-q4k-ehf16-af32',
      'qwen-3-5-0-8b-q4k-ehaf16',
      'qwen-3-5-2b-q4k-ehaf16',
    ],
    'public demo selector should contain the six intended browser models'
  );
  for (const entry of selected) {
    assert.equal(entry.lifecycle?.status?.tested, 'verified', `${entry.modelId} must be verified`);
    assert.ok(entry.demoLabel, `${entry.modelId} must have a compact demo label`);
    assert.ok(entry.demoRole, `${entry.modelId} must explain its demo role`);
  }
  const translate = selected.find((entry) => entry.family === 'translategemma');
  assert.deepEqual(translate.demoWarningBadges ?? [], []);
  assert.equal(translate.demoWarningText ?? '', '');
}

{
  const catalog = [
    { modelId: 'remote-only' },
    {
      modelId: 'visible-af16',
      demoFallbackVariant: { modelId: 'stored-af32' },
    },
    { modelId: 'stored-qwen' },
  ];
  const registrations = [
    { modelId: 'unlisted-model', savedAtUtc: '2026-06-01T00:00:00.000Z' },
    { modelId: 'stored-af32', savedAtUtc: '2026-06-02T00:00:00.000Z' },
    { modelId: 'stored-qwen', savedAtUtc: '2026-06-03T00:00:00.000Z' },
  ];

  assert.deepEqual(
    selectDefaultStoredModel(catalog, registrations),
    { modelId: 'stored-qwen', displayModelId: 'stored-qwen' },
    'startup should choose the most recently saved visible model instead of a remote default'
  );
  assert.deepEqual(
    selectDefaultStoredModel(catalog, registrations, 'stored-af32'),
    { modelId: 'stored-af32', displayModelId: 'visible-af16' },
    'startup should reuse the last-used stored execution lane and activate its visible card'
  );
  assert.equal(
    selectDefaultStoredModel(catalog, [{ modelId: 'unlisted-model' }]),
    null,
    'startup should not select a stored model that is absent from the visible demo catalog'
  );
}

{
  const selected = selectDemoCatalogEntries([
    {
      ...DEMO_READY,
      modelId: 'experimental-enes-translator',
      label: 'Experimental EN/ES Translator',
      quickstart: false,
      demoVisible: true,
      modes: ['translate'],
      weightPackId: 'experimental-enes-translator-wp-v1',
      demoWarningBadges: ['Experimental', 'EN ↔ ES only'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/experimental-enes-translator',
      },
    },
  ]);

  assert.deepEqual(
    selected.map((entry) => entry.modelId),
    ['experimental-enes-translator'],
    'demo catalog should include explicitly visible translate-only models'
  );
  assert.deepEqual(
    selected[0].demoWarningBadges,
    ['Experimental', 'EN ↔ ES only'],
    'translate-only demo cards should preserve their experimental scope badges'
  );
}

{
  const selected = selectDemoCatalogEntries([
    {
      ...DEMO_READY,
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
      label: 'Gemma 4 E2B (Q4K/F32a/INT4 PLE)',
      weightPackId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple-wp-catalog-v1',
      demoPreferredVariantId: 'gemma-4-e2b-it-q4k-ehf16-af16-int4ple',
      quickstart: true,
      demoVisible: true,
      modes: ['text', 'vision'],
      sortOrder: 14,
    },
    {
      ...DEMO_WEIGHTS_REF_READY,
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af16-int4ple',
      label: 'Gemma 4 E2B (Q4K/F16a/INT4 PLE)',
      weightPackId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple-wp-catalog-v1',
      quickstart: false,
      demoVisible: false,
      modes: ['text', 'vision'],
      sortOrder: 15,
    },
  ], {
    localBaseUrls: new Map([
      ['gemma-4-e2b-it-q4k-ehf16-af32-int4ple', 'http://localhost:8080/models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple'],
      ['gemma-4-e2b-it-q4k-ehf16-af16-int4ple', 'http://localhost:8080/models/local/gemma-4-e2b-it-q4k-ehf16-af16-int4ple'],
    ]),
  });

  assert.deepEqual(
    selected.map((entry) => entry.modelId),
    ['gemma-4-e2b-it-q4k-ehf16-af16-int4ple'],
    'Gemma 4 E2B demo card should prefer the af16 weights-ref sibling when it is reachable'
  );
  assert.equal(
    selected[0].demoFallbackVariant?.modelId,
    'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    'Gemma 4 E2B surfaced af16 card carries the af32 INT4-PLE primary as fallback'
  );
}

{
  const selected = selectDemoCatalogEntries([
    {
      ...DEMO_READY,
      modelId: 'missing-source-text-model',
      quickstart: true,
      demoVisible: true,
      modes: ['text'],
      sortOrder: 20,
    },
    {
      ...DEMO_READY,
      modelId: 'hf-backed-text-model',
      quickstart: true,
      demoVisible: true,
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
      label: 'Gemma 4 31B (Q4K)',
      weightPackId: 'gemma-4-31b-it-text-q4k-ehf16-af32-wp-catalog-v1',
      demoPreferredVariantId: 'gemma-4-31b-it-text-q4k-ehf16-af16',
      quickstart: false,
      demoVisible: true,
      modes: ['text'],
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/gemma-4-31b-it-text-q4k-ehf16-af32',
      },
      sortOrder: 16,
      sizeBytes: 19295019640,
      demoWarningBadges: ['19.3 GB / High RAM'],
      demoWarningText: 'Large download. Use a high-RAM machine.',
    },
    {
      ...DEMO_WEIGHTS_REF_READY,
      modelId: 'gemma-4-31b-it-text-q4k-ehf16-af16',
      label: 'Gemma 4 31B (Q4K/F16a)',
      weightPackId: 'gemma-4-31b-it-text-q4k-ehf16-af32-wp-catalog-v1',
      quickstart: false,
      demoVisible: false,
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
      'gemma-4-31b-it-text-q4k-ehf16-af16',
    ],
    'demo catalog should surface the af16 sibling as the visible card when the primary opts in'
  );
  assert.equal(
    selected[0].label,
    'Gemma 4 31B (Q4K/F16a)',
    'surfaced card preserves the af16 lane label'
  );
  assert.equal(
    selected[0].demoFallbackVariant?.modelId,
    'gemma-4-31b-it-text-q4k-ehf16-af32',
    'surfaced af16 card carries the af32 primary as demoFallbackVariant'
  );
  assert.equal(
    selected[0].weightsRefPrimary,
    'gemma-4-31b-it-text-q4k-ehf16-af32',
    'surfaced af16 card names its af32 primary in weightsRefPrimary'
  );
  assert.equal(
    selected[0].demoVisible,
    true,
    'surfaced af16 card is forced visible regardless of its own demoVisible flag'
  );
  assert.deepEqual(
    selected[0].demoWarningBadges,
    ['19.3 GB / High RAM'],
    'surfaced af16 card inherits warning badges from the primary when the af16 is silent'
  );
  assert.equal(
    selected[0].demoWarningText,
    'Large download. Use a high-RAM machine.',
    'surfaced af16 card inherits warning text from the primary when the af16 is silent'
  );
  assert.equal(
    selected[0].sizeBytes,
    19295019640,
    'surfaced af16 card inherits sizeBytes from the primary'
  );
}

{
  const selected = selectDemoCatalogEntries([
    {
      ...DEMO_READY,
      modelId: 'gemma-4-31b-it-text-q4k-ehf16-af32',
      weightPackId: 'gemma-4-31b-it-text-q4k-ehf16-af32-wp-catalog-v1',
      demoPreferredVariantId: 'gemma-4-31b-it-text-q4k-ehf16-af16',
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
      lifecycle: {
        status: {
          runtime: 'experimental',
          tested: 'none',
        },
      },
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
    ['gemma-4-31b-it-text-q4k-ehf16-af32'],
    'demo catalog falls back to the af32 primary when the af16 sibling is unverified'
  );
  assert.equal(
    selected[0].demoFallbackVariant,
    null,
    'demo catalog must not attach an unverified af16 as the af32 fallback either'
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
  const entry = {
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af16-int4ple',
    demoFallbackVariant: {
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    },
  };
  const manifests = new Map([
    ['gemma-4-e2b-it-q4k-ehf16-af16-int4ple', readManifest('gemma-4-e2b-it-q4k-ehf16-af16-int4ple')],
    ['gemma-4-e2b-it-q4k-ehf16-af32-int4ple', readManifest('gemma-4-e2b-it-q4k-ehf16-af32-int4ple')],
  ]);
  const selected = selectDemoExecutionEntryForCapabilities(entry, manifests, {
    hasF16: true,
    hasSubgroups: true,
    hasSubgroupsF16: true,
    maxWorkgroupStorageSize: 32768,
    adapterInfo: {
      vendor: 'apple',
      architecture: 'metal-3',
    },
  });
  assert.equal(
    selected.modelId,
    'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    'Apple Metal should use the af32 E2B fallback when capability rules reject the af16 manifest'
  );
}

{
  const manifests = new Map([
    ['qwen-3-5-0-8b-q4k-ehaf16', readManifest('qwen-3-5-0-8b-q4k-ehaf16')],
  ]);
  assert.throws(
    () => selectDemoExecutionEntryForCapabilities({
      modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    }, manifests, {
      hasF16: true,
      hasSubgroups: false,
      hasSubgroupsF16: false,
      maxWorkgroupStorageSize: 32768,
      adapterInfo: {
        vendor: 'apple',
        architecture: 'safari',
      },
    }),
    /Hybrid linear-attention model "qwen-3-5-0-8b-q4k-ehaf16" cannot apply a global non-Q4/,
    'Safari/no-subgroup Qwen 3.5 must fail before download instead of reaching model load'
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
    'Downloaded · 612 MB'
  );
  assert.equal(
    buildModelCardDetail({ sizeBytes: 1536 * 1024 * 1024 }, 'loaded'),
    'Loaded · 1.5 GB'
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
    'Downloaded · Shared with gemma-4-31b-it-text-q4k-ehf16-af32'
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

// Surfaced af16 cards nest the af32 primary inside demoFallbackVariant.
// findPrimaryForWeightPack must walk that shape so download/load still
// resolves the primary against the post-selection catalog view.
{
  const surfacedCatalog = [
    {
      modelId: 'surfaced-af16',
      weightPackId: 'wp-1',
      artifactCompleteness: 'weights-ref',
      weightsRefAllowed: true,
      demoFallbackVariant: {
        modelId: 'nested-primary',
        weightPackId: 'wp-1',
        artifactCompleteness: 'complete',
        weightsRefAllowed: false,
      },
    },
  ];
  assert.equal(
    findPrimaryForWeightPack(surfacedCatalog, 'wp-1')?.modelId,
    'nested-primary',
    'findPrimaryForWeightPack must descend into demoFallbackVariant'
  );
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
