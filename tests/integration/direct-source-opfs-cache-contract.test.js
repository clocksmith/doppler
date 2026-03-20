import assert from 'node:assert/strict';

import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import {
  cleanup,
  computeSHA256,
  deleteFileFromStore,
  loadManifestFromStore,
  openModelStore,
} from '../../src/storage/shard-manager.js';
import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { downloadModel } from '../../src/storage/downloader.js';
import {
  createStoredSourceArtifactContext,
  verifyStoredSourceArtifact,
} from '../../src/storage/source-artifact-store.js';
import { ensureModelCached } from '../../src/tooling/opfs-cache.js';
import { buildSourceRuntimeBundle, DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE } from '../../src/tooling/source-runtime-bundle.js';

const originalFetch = globalThis.fetch;
const originalNavigator = globalThis.navigator;

async function createDirectSourceManifest() {
  const sourceBytes = new Uint8Array([0, 0, 128, 63]);
  const configText = JSON.stringify({ model_type: 'gemma3_text', architectures: ['Gemma3ForCausalLM'] });
  const tokenizerJson = { version: '1.0', model: { vocab: [['<bos>', 0]] } };
  const tokenizerText = JSON.stringify(tokenizerJson);

  const sourceHash = await computeSHA256(sourceBytes);
  const configHash = await computeSHA256(new TextEncoder().encode(configText));
  const tokenizerHash = await computeSHA256(new TextEncoder().encode(tokenizerText));

  const bundle = await buildSourceRuntimeBundle({
    modelId: 'gemma-3-direct-source-opfs-test',
    modelName: 'gemma-3-direct-source-opfs-test',
    modelType: 'transformer',
    sourceKind: 'safetensors',
    architectureHint: 'gemma3',
    architecture: {
      numLayers: 1,
      hiddenSize: 1,
      intermediateSize: 4,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDim: 1,
      vocabSize: 8,
      maxSeqLen: 16,
    },
    rawConfig: {
      model_type: 'gemma3_text',
      architectures: ['Gemma3ForCausalLM'],
    },
    inference: {
      ...DEFAULT_MANIFEST_INFERENCE,
    },
    tensors: [
      {
        name: 'model.embed_tokens.weight',
        shape: [1],
        dtype: 'F32',
        size: sourceBytes.byteLength,
        offset: 0,
        sourcePath: 'weights/model.safetensors',
      },
    ],
    sourceFiles: [
      {
        path: 'weights/model.safetensors',
        size: sourceBytes.byteLength,
        hash: sourceHash,
        hashAlgorithm: 'sha256',
      },
    ],
    auxiliaryFiles: [
      {
        path: 'config.json',
        size: new TextEncoder().encode(configText).byteLength,
        hash: configHash,
        hashAlgorithm: 'sha256',
        kind: 'config',
      },
      {
        path: 'tokenizer.json',
        size: new TextEncoder().encode(tokenizerText).byteLength,
        hash: tokenizerHash,
        hashAlgorithm: 'sha256',
        kind: 'tokenizer_json',
      },
    ],
    hashAlgorithm: 'sha256',
    tokenizerJson,
    tokenizerJsonPath: 'tokenizer.json',
    eosTokenId: 1,
  });

  bundle.manifest.metadata.sourceRuntime.pathSemantics = DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE;

  return {
    manifest: bundle.manifest,
    sourceBytes,
    configText,
    tokenizerJson,
    tokenizerText,
  };
}

try {
  setRuntimeConfig({
    loading: {
      storage: {
        backend: {
          backend: 'memory',
          memory: {
            maxBytes: 1024 * 1024,
          },
        },
      },
    },
  });

  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        getDirectory() {},
        estimate: async () => ({ usage: 0, quota: 1024 * 1024 }),
        persisted: async () => true,
        persist: async () => true,
      },
    },
    configurable: true,
  });

  const fixture = await createDirectSourceManifest();
  const baseUrl = 'https://example.test/direct-source';

  globalThis.fetch = async (url) => {
    const href = String(url);
    if (href === `${baseUrl}/manifest.json`) {
      return new Response(JSON.stringify(fixture.manifest), { status: 200 });
    }
    if (href === `${baseUrl}/weights/model.safetensors`) {
      return new Response(fixture.sourceBytes, { status: 200 });
    }
    if (href === `${baseUrl}/config.json`) {
      return new Response(fixture.configText, { status: 200 });
    }
    if (href === `${baseUrl}/tokenizer.json`) {
      return new Response(fixture.tokenizerText, { status: 200 });
    }
    throw new Error(`Unexpected fetch: ${href}`);
  };

  const downloaded = await downloadModel(baseUrl);
  assert.equal(downloaded, true);

  await openModelStore(fixture.manifest.modelId);
  const manifestText = await loadManifestFromStore();
  assert.ok(manifestText);
  const storedManifest = JSON.parse(manifestText);

  const integrity = await verifyStoredSourceArtifact(storedManifest, { checkHashes: true });
  assert.equal(integrity.valid, true);
  assert.deepEqual(integrity.missingFiles, []);
  assert.deepEqual(integrity.corruptFiles, []);

  const storageContext = createStoredSourceArtifactContext(storedManifest, { verifyHashes: true });
  const shardBuffer = await storageContext.loadShard(0);
  assert.deepEqual(new Uint8Array(shardBuffer), fixture.sourceBytes);
  assert.equal(storageContext.loadShardRange, null);
  assert.equal(storageContext.streamShardRange, null);
  const fastStorageContext = createStoredSourceArtifactContext(storedManifest, { verifyHashes: false });
  const shardRange = await fastStorageContext.loadShardRange(0, 1, 2);
  assert.deepEqual(Array.from(new Uint8Array(shardRange)), [0, 128]);
  const streamed = [];
  for await (const chunk of fastStorageContext.streamShardRange(0, 0, 4, { chunkBytes: 2 })) {
    streamed.push(...chunk);
  }
  assert.deepEqual(streamed, Array.from(fixture.sourceBytes));
  assert.deepEqual(await storageContext.loadTokenizerJson(), fixture.tokenizerJson);

  const cached = await ensureModelCached(fixture.manifest.modelId, baseUrl);
  assert.equal(cached.cached, true);
  assert.equal(cached.fromCache, true);

  await deleteFileFromStore('tokenizer.json');
  const missing = await ensureModelCached(fixture.manifest.modelId, baseUrl);
  assert.equal(missing.cached, true);
  assert.equal(missing.fromCache, false);
  const repaired = await verifyStoredSourceArtifact(storedManifest, { checkHashes: true });
  assert.equal(repaired.valid, true);
} finally {
  globalThis.fetch = originalFetch;
  if (originalNavigator === undefined) {
    delete globalThis.navigator;
  } else {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
    });
  }
  resetRuntimeConfig();
  await cleanup();
}

console.log('direct-source-opfs-cache-contract.test: ok');
