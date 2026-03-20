import assert from 'node:assert/strict';

import {
  buildHostedRegistryPayload,
  buildHfResolveUrl,
  buildManifestUrl,
  buildPublishedRegistryEntry,
  buildShardUrl,
  collectDuplicateModelIds,
  extractCommitShaFromUrl,
  fetchRepoHeadSha,
  isHostedRegistryApprovedEntry,
  resolveDemoRegistryEntryBaseUrl,
  shouldDemoSurfaceRemoteRegistryEntry,
  validateLocalHfEntryShape,
} from '../../src/tooling/hf-registry-utils.js';

{
  assert.equal(
    buildHfResolveUrl('Clocksmith/rdrr', 'abc123', 'models/demo-model'),
    'https://huggingface.co/Clocksmith/rdrr/resolve/abc123/models/demo-model'
  );
  assert.equal(buildManifestUrl('https://host/models/demo-model/'), 'https://host/models/demo-model/manifest.json');
  assert.equal(
    buildShardUrl('https://host/models/demo-model', { filename: 'shard_00000.bin' }),
    'https://host/models/demo-model/shard_00000.bin'
  );
}

{
  const entry = {
    modelId: 'translategemma-4b-it-q4k-ehf16-af32',
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: null,
      path: 'models/translategemma-4b-it-q4k-ehf16-af32',
    },
    lifecycle: {
      availability: {
        hf: false,
      },
    },
  };
  const next = buildPublishedRegistryEntry(entry, 'f3bac04695d52b4bf075d23ce094a7b73c17a913');
  assert.equal(next.hf.revision, 'f3bac04695d52b4bf075d23ce094a7b73c17a913');
  assert.equal(next.lifecycle.availability.hf, true);
}

{
  assert.throws(
    () => buildPublishedRegistryEntry({
      modelId: 'missing-hf-fields',
      hf: {
        revision: null,
      },
    }, 'f3bac04695d52b4bf075d23ce094a7b73c17a913'),
    /requires explicit hf\.repoId/
  );
  assert.throws(
    () => buildPublishedRegistryEntry({
      modelId: 'missing-hf-path',
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: null,
        path: '',
      },
    }, 'f3bac04695d52b4bf075d23ce094a7b73c17a913'),
    /requires explicit hf\.path/
  );
}

{
  const bad = {
    modelId: 'bad-model',
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: null,
      path: '',
    },
  };
  assert.deepEqual(validateLocalHfEntryShape(bad), [
    'bad-model: hf.revision is required when lifecycle.availability.hf=true',
    'bad-model: hf.path is required when lifecycle.availability.hf=true',
  ]);
}

{
  assert.equal(isHostedRegistryApprovedEntry({
    lifecycle: {
      availability: {
        hf: true,
      },
      status: {
        runtime: 'active',
        tested: 'verified',
      },
    },
  }), true);
  assert.equal(isHostedRegistryApprovedEntry({
    lifecycle: {
      availability: {
        hf: true,
      },
      status: {
        runtime: 'active',
        tested: 'failing',
      },
    },
  }), false);
}

{
  const remoteHfEntry = {
    modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: 'cd6c12be0e83e92d6dbd92598a0aa94391ec7e94',
      path: 'models/gemma-3-270m-it-q4k-ehf16-af32',
    },
    baseUrl: './local/gemma-3-270m-it-q4k-ehf16-af32',
  };
  assert.equal(shouldDemoSurfaceRemoteRegistryEntry(remoteHfEntry, 'https://huggingface.co/Clocksmith/rdrr/resolve/main/registry/catalog.json'), true);
  assert.equal(
    resolveDemoRegistryEntryBaseUrl(remoteHfEntry, 'https://huggingface.co/Clocksmith/rdrr/resolve/main/registry/catalog.json'),
    'https://huggingface.co/Clocksmith/rdrr/resolve/cd6c12be0e83e92d6dbd92598a0aa94391ec7e94/models/gemma-3-270m-it-q4k-ehf16-af32'
  );
}

{
  const nonFetchableRemoteEntry = {
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: null,
      path: 'models/qwen-3-5-0-8b-q4k-ehaf16',
    },
    baseUrl: null,
  };
  assert.equal(shouldDemoSurfaceRemoteRegistryEntry(nonFetchableRemoteEntry, 'https://huggingface.co/Clocksmith/rdrr/resolve/main/registry/catalog.json'), false);
}

{
  const payload = buildHostedRegistryPayload({
    version: 1,
    lifecycleSchemaVersion: 1,
    updatedAt: '2026-03-11',
    models: [
      {
        modelId: 'failing-qwen',
        sortOrder: 2,
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
          path: 'models/failing-qwen',
        },
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            runtime: 'active',
            tested: 'failing',
          },
        },
      },
      {
        modelId: 'verified-gemma',
        sortOrder: 1,
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
          path: 'models/verified-gemma',
        },
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            runtime: 'active',
            tested: 'verified',
          },
        },
      },
    ],
  });
  assert.deepEqual(payload.models.map((entry) => entry.modelId), ['verified-gemma']);
}

{
  assert.equal(
    extractCommitShaFromUrl('https://huggingface.co/Clocksmith/rdrr/commit/f3bac04695d52b4bf075d23ce094a7b73c17a913'),
    'f3bac04695d52b4bf075d23ce094a7b73c17a913'
  );
  assert.deepEqual(
    collectDuplicateModelIds([
      { modelId: 'a' },
      { modelId: 'b' },
      { modelId: 'a' },
    ]),
    ['a']
  );
}

{
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ sha: 'f3bac04695d52b4bf075d23ce094a7b73c17a913' }),
  });
  try {
    assert.equal(
      await fetchRepoHeadSha('Clocksmith/rdrr'),
      'f3bac04695d52b4bf075d23ce094a7b73c17a913'
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
}

console.log('hf-registry-utils.test: ok');
