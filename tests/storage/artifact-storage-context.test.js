import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';

import {
  ARTIFACT_FORMAT_DIRECT_SOURCE,
  ARTIFACT_FORMAT_RDRR,
  createArtifactStorageContext,
  createHttpArtifactStorageContext,
  createNodeFileArtifactStorageContext,
  getArtifactFormat,
} from '../../src/storage/artifact-storage-context.js';

// Independent official BLAKE3 digest, plus the retained pre-correction digest
// for identical bytes. Verification must reject the latter without mutation.
{
  const vectors = JSON.parse(readFileSync(new URL('./blake3-vectors.json', import.meta.url), 'utf8'));
  const standard = vectors.cases.find(row => row.length === 65).hex;
  const legacy = 'b5ef4905c6aa739c7c0573dc76813e0b6ef2da83aafbc212bb4e30181783671f';
  const bytes = Uint8Array.from({ length: 65 }, (_, index) => index);
  for (const streaming of [false, true]) {
    for (const digest of [standard, legacy]) {
      const manifest = { modelId: 'blake3-control', hashAlgorithm: 'blake3', shards: [
        { filename: 'weights.bin', size: bytes.length, hash: digest },
      ] };
      const original = JSON.stringify(manifest);
      const context = createArtifactStorageContext({
        manifest, verifyHashes: true,
        async readRange(_path, offset, length) { return bytes.slice(offset, offset + length); },
        streamRange: streaming ? async function* () {
          yield bytes.subarray(0, 64);
          yield bytes.subarray(64);
        } : undefined,
      });
      if (digest === standard) {
        assert.deepEqual(new Uint8Array(await context.loadShard(0)), bytes);
      } else {
        await assert.rejects(context.loadShard(0), /Artifact shard hash mismatch/);
        await assert.rejects(context.loadShard(0), /Artifact shard hash mismatch/);
      }
      assert.equal(JSON.stringify(manifest), original, 'verification must not rewrite legacy identity');
      assert.deepEqual([...bytes], Array.from({ length: 65 }, (_, index) => index));
    }
  }
}

{
  const directSourceManifest = {
    modelId: 'direct-source-format-test',
    shards: [{ filename: 'source_00000.bin', size: 16 }],
    metadata: {
      sourceRuntime: {
        mode: 'direct-source',
        sourceKind: 'safetensors',
      },
    },
  };
  assert.equal(getArtifactFormat(directSourceManifest), ARTIFACT_FORMAT_DIRECT_SOURCE);
}

{
  const rdrrManifest = {
    modelId: 'rdrr-format-test',
    shards: [{ filename: 'shard_00000.bin', size: 16 }],
  };
  assert.equal(getArtifactFormat(rdrrManifest), ARTIFACT_FORMAT_RDRR);
}

{
  const storedRdrrManifest = {
    modelId: 'stored-rdrr-format-test',
    shards: [{ filename: 'shard_00000.bin', size: 16 }],
    metadata: {
      sourceRuntime: {
        mode: 'direct-source',
        sourceKind: 'rdrr',
      },
    },
  };
  assert.equal(getArtifactFormat(storedRdrrManifest), ARTIFACT_FORMAT_RDRR);
}

{
  assert.equal(getArtifactFormat({ modelId: 'invalid-format-test' }), null);
}

{
  const fixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-artifact-storage-context-'));
  const shardPath = path.join(fixtureDir, 'source_00000.bin');
  const manifest = {
    modelId: 'artifact-node-file-reader-cache-test',
    shards: [{ filename: 'source_00000.bin', size: 32 }],
  };
  writeFileSync(shardPath, Uint8Array.from({ length: 32 }, (_, index) => index));
  let storageContext = null;
  try {
    storageContext = createNodeFileArtifactStorageContext(fixtureDir, manifest);
    assert.ok(storageContext, 'node file artifact storage context should be created for filesystem manifests');
    assert.equal(typeof storageContext.close, 'function');
    const firstRange = await storageContext.loadShardRange(0, 0, 8);
    const secondRange = await storageContext.loadShardRange(0, 8, 8);
    assert.equal(new Uint8Array(firstRange).byteLength, 8);
    assert.equal(new Uint8Array(secondRange).byteLength, 8);
    await storageContext.close?.();
  } finally {
    await storageContext?.close?.();
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const calls = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    return {
      ok: true,
      status: 200,
      arrayBuffer: async () => Uint8Array.from([9, 8]).buffer,
      text: async () => '{"ok":true}',
    };
  };
  try {
    const manifest = {
      modelId: 'artifact-http-no-store-test',
      hashAlgorithm: 'blake3',
      tensorsFile: 'tensors.json',
      shards: [{ filename: 'shard_00000.bin', size: 4, hash: null }],
    };
    const storageContext = createHttpArtifactStorageContext('https://example.com/model', manifest, {
      verifyHashes: false,
    });
    const bytes = new Uint8Array(await storageContext.loadShardRange(0, 1, 2));
    assert.deepEqual(Array.from(bytes), [9, 8]);
    assert.equal(await storageContext.loadTensorsJson(), '{"ok":true}');
    assert.equal(calls[0].url, 'https://example.com/model/shard_00000.bin');
    assert.equal(calls[0].init.cache, 'no-store');
    assert.equal(calls[0].init.headers.Range, 'bytes=1-2');
    assert.equal(calls[1].url, 'https://example.com/model/tensors.json');
    assert.equal(calls[1].init.cache, 'no-store');
  } finally {
    globalThis.fetch = originalFetch;
  }
}

{
  const calls = [];
  const originalFetch = globalThis.fetch;
  const fullShard = Uint8Array.from({ length: 16 }, (_, index) => index);
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    return {
      ok: true,
      status: 200,
      arrayBuffer: async () => fullShard.buffer.slice(0),
      text: async () => '{"ok":true}',
    };
  };
  try {
    const manifest = {
      modelId: 'artifact-http-ignored-range-test',
      hashAlgorithm: 'blake3',
      shards: [{ filename: 'shard_00000.bin', size: 16, hash: null }],
    };
    const storageContext = createHttpArtifactStorageContext('https://example.com/model', manifest, {
      verifyHashes: false,
    });
    const bytes = new Uint8Array(await storageContext.loadShardRange(0, 5, 4));
    assert.deepEqual(Array.from(bytes), [5, 6, 7, 8]);
    assert.equal(bytes.byteOffset, 0);
    assert.equal(calls[0].init.headers.Range, 'bytes=5-8');
  } finally {
    globalThis.fetch = originalFetch;
  }
}

{
  const calls = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    const range = String(init?.headers?.Range || '');
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    assert.ok(match, `expected range header, got ${range}`);
    const start = Number(match[1]);
    const end = Number(match[2]);
    return {
      ok: true,
      status: 206,
      arrayBuffer: async () => Uint8Array.from(
        { length: end - start + 1 },
        (_, index) => start + index
      ).buffer,
      text: async () => '{"ok":true}',
    };
  };
  try {
    const manifest = {
      modelId: 'artifact-http-range-block-cache-test',
      hashAlgorithm: 'blake3',
      shards: [{ filename: 'shard_00000.bin', size: 16, hash: null }],
    };
    const storageContext = createHttpArtifactStorageContext('https://example.com/model', manifest, {
      verifyHashes: false,
      rangeCacheBlockBytes: 4,
      rangeCacheMaxBytes: 8,
      rangeCacheMinBytes: 2,
    });
    const first = new Uint8Array(await storageContext.loadShardRange(0, 1, 2));
    const second = new Uint8Array(await storageContext.loadShardRange(0, 2, 2));
    assert.deepEqual(Array.from(first), [1, 2]);
    assert.deepEqual(Array.from(second), [2, 3]);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].init.cache, 'no-store');
    assert.equal(calls[0].init.headers.Range, 'bytes=0-3');
  } finally {
    globalThis.fetch = originalFetch;
  }
}

console.log('artifact-storage-context.test: ok');
