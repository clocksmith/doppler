import assert from 'node:assert/strict';

import {
  buildHfResolveUrl,
  buildManifestUrl,
  buildPublishedRegistryEntry,
  buildShardUrl,
  collectDuplicateModelIds,
  extractCommitShaFromUrl,
  resolveDemoRegistryEntryBaseUrl,
  shouldDemoSurfaceRemoteRegistryEntry,
  validateLocalHfEntryShape,
} from '../../tools/hf-registry-utils.js';

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
  const remoteHfEntry = {
    modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: '4efe64a914892e98be50842aeb16c3b648cc68a5',
      path: 'models/gemma-3-270m-it-q4k-ehf16-af32',
    },
    baseUrl: './local/gemma-3-270m-it-q4k-ehf16-af32',
  };
  assert.equal(shouldDemoSurfaceRemoteRegistryEntry(remoteHfEntry, 'https://huggingface.co/Clocksmith/rdrr/resolve/main/registry/catalog.json'), true);
  assert.equal(
    resolveDemoRegistryEntryBaseUrl(remoteHfEntry, 'https://huggingface.co/Clocksmith/rdrr/resolve/main/registry/catalog.json'),
    'https://huggingface.co/Clocksmith/rdrr/resolve/4efe64a914892e98be50842aeb16c3b648cc68a5/models/gemma-3-270m-it-q4k-ehf16-af32'
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

console.log('hf-registry-utils.test: ok');
