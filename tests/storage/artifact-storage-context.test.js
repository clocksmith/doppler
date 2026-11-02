import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';

import {
  ARTIFACT_FORMAT_DIRECT_SOURCE,
  ARTIFACT_FORMAT_RDRR,
  createArtifactStorageContext,
  createHttpArtifactStorageContext,
  createNodeFileArtifactStorageContext,
  getArtifactFormat,
} from '../../src/storage/artifact-storage-context.js';

// Whole-file consumers receive the exact bytes that passed verification, in
// one read. Each full read remains verified even when its source later changes.
{
  const original = Uint8Array.from({ length: 73 }, (_, index) => index);
  let stored = original;
  const calls = [];
  const context = createArtifactStorageContext({
    manifest: { modelId: 'full-shard-verification', hashAlgorithm: 'sha256', shards: [
      { filename: 'weights.bin', size: original.length,
        hash: createHash('sha256').update(original).digest('hex') },
    ] },
    verifyHashes: true,
    async readRange(path, offset, length) {
      calls.push({ path, offset, length });
      return stored.slice(offset, offset + length);
    },
    async *streamRange() { throw new Error('Full reads must not stream a second copy'); },
  });
  assert.deepEqual(new Uint8Array(await context.loadShard(0)), original);
  assert.deepEqual(calls, [{ path: 'weights.bin', offset: 0, length: original.length }]);
  stored = original.slice();
  stored[5] ^= 1;
  await assert.rejects(context.loadShard(0), /Artifact shard hash mismatch/);
  stored = original.subarray(0, original.length - 1);
  await assert.rejects(context.loadShard(0), /Artifact shard short read for verification/);
  stored = original;
  assert.deepEqual(new Uint8Array(await context.loadShard(0)), original);
}

// Partial consumers continue to verify incrementally before returning ranges.
for (const verificationBufferBytes of [undefined, 0, 72]) {
  const bytes = Uint8Array.from({ length: 73 }, (_, index) => index);
  let streamed = 0;
  const context = createArtifactStorageContext({
    manifest: { modelId: 'range-shard-verification', hashAlgorithm: 'sha256', shards: [
      { filename: 'weights.bin', size: bytes.length,
        hash: createHash('sha256').update(bytes).digest('hex') },
    ] },
    verifyHashes: true,
    verificationBufferBytes,
    async readRange(_path, offset, length) { return bytes.slice(offset, offset + length); },
    async *streamRange() {
      streamed++;
      yield bytes.subarray(0, 32);
      yield bytes.subarray(32);
    },
  });
  assert.deepEqual(new Uint8Array(await context.loadShardRange(0, 5, 3)), bytes.slice(5, 8));
  assert.deepEqual(new Uint8Array(await context.loadShardRange(0, 8, 3)), bytes.slice(8, 11));
  assert.equal(streamed, 1, 'Verified ranges share one bounded verification task');
}

// The optional whole-source verification workspace is shared across concurrent
// range consumers, and a failed verification does not block another source.
{
  const bytes = Uint8Array.from({ length: 73 }, (_, index) => index);
  const hash = createHash('sha256').update(bytes).digest('hex');
  const manifest = { modelId: 'buffered-range-verification', hashAlgorithm: 'sha256', shards: [
    { filename: 'first.bin', size: bytes.length, hash },
    { filename: 'second.bin', size: bytes.length, hash },
  ] };
  let active = 0, maximumActive = 0, failures = 1, fullReads = 0;
  const options = {
    manifest, verifyHashes: true, verificationBufferBytes: bytes.length,
    async readRange(path, offset, length) {
      if (length === bytes.length) {
        fullReads++;
        active++;
        maximumActive = Math.max(maximumActive, active);
        await Promise.resolve();
        active--;
        if (path === 'first.bin' && failures-- > 0) return bytes.subarray(0, bytes.length - 1);
      }
      return bytes.slice(offset, offset + length);
    },
    async *streamRange() { throw new Error('The admitted buffer must be verified as a whole'); },
  };
  const context = createArtifactStorageContext(options);
  const results = await Promise.allSettled([
    context.loadShardRange(0, 5, 3), context.loadShardRange(1, 8, 3),
  ]);
  assert.equal(results[0].status, 'rejected');
  assert.match(String(results[0].reason), /short read/);
  assert.equal(results[1].status, 'fulfilled');
  assert.deepEqual(new Uint8Array(results[1].value), bytes.slice(8, 11));
  assert.deepEqual(new Uint8Array(await context.loadShardRange(0, 5, 3)), bytes.slice(5, 8));
  assert.deepEqual(new Uint8Array(await context.loadShardRange(0, 8, 3)), bytes.slice(8, 11));
  assert.equal(fullReads, 3, 'Failed source retries; successful verification is reused');
  assert.equal(maximumActive, 1);
  for (const value of [-1, 1.5, NaN, null, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => createArtifactStorageContext({ ...options, verificationBufferBytes: value }), /non-negative safe integer/);
  }
}

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
  assert.throws(() => createArtifactStorageContext({
    manifest: directSourceManifest, verificationBufferBytes: 16,
    async readRange() { throw new Error('Unsupported policy must fail before reading'); },
  }), /supported only for RDRR/);
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
