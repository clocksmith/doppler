import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { createNodeConvertIO, createFileRangeReader } from '../../src/tooling/node-converter/output.js';
import { appendRetainedSourceShards } from '../../src/converter/retained-source-shards.js';
import { convertModel } from '../../src/converter/core.js';
import { createConverterConfig } from '../../src/config/schema/converter.schema.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';

// Filesystems without links must still admit in-root source rows, and reject an
// unsupported layout before doing any tensor conversion or shard writes.
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-retention-'));
const sourceDir = path.join(dir, 'source');
const outputDir = path.join(dir, 'output');
await fs.mkdir(sourceDir); await fs.mkdir(outputDir);
const bytes = Uint8Array.from({ length: 32 }, (_, i) => i);
const sourcePath = path.join(sourceDir, 'table.safetensors');
await fs.writeFile(sourcePath, bytes);
const pin = createHash('sha256').update(bytes).digest('hex');
const retained = [{ name: 'table', sourcePath, offset: 0, size: 32, shape: [4, 4], dtype: 'BF16',
  role: 'ngram_table', group: 'ngram_table', retainedSource: { file: 'table.safetensors', sha256: pin } }];
const reader = createFileRangeReader();
const options = { hashAlgorithm: 'sha256', computeHash: async () => pin, readRange: reader.readRange };
const originalLink = fs.link;
let linkCalls = 0;
fs.link = async () => { linkCalls++; throw Object.assign(new Error('link unavailable'), { code: 'EPERM' }); };
try {
  const io = createNodeConvertIO(dir, options);
  await io.preflightRetainedSources([sourcePath]);
  const shards = [], tensors = {};
  await appendRetainedSourceShards(retained, io, shards, tensors, { hashAlgorithm: 'sha256', manifestConfig: {} });
  assert.equal(linkCalls, 0, 'an existing source inside the artifact needs no new filesystem link');
  assert.equal(shards[0].filename, 'source/table.safetensors');
  assert.deepEqual(new Uint8Array(await io.readShardRange(0, 4, 8)), bytes.slice(4, 12));
  const storage = createNodeFileArtifactStorageContext(pathToFileURL(dir).href,
    { modelId: 'retention-test', hashAlgorithm: 'sha256', shards, tensors });
  try { assert.deepEqual(new Uint8Array(await storage.loadShardRange(0, 8, 8)), bytes.slice(8, 16)); }
  finally { await storage.close(); }
  await assert.rejects(io.retainSourceFile(sourcePath, 1, '0'.repeat(64)), /differs from its pin/);
  assert.deepEqual(new Uint8Array(await fs.readFile(sourcePath)), bytes, 'a bad pin never deletes the original source');

  const unsupported = createNodeConvertIO(outputDir, options);
  let reads = 0, writes = 0;
  await assert.rejects(convertModel({ name: 'preflight-test', tensors: [
    { name: 'model.embed_tokens.weight', shape: [2, 2], dtype: 'F32', size: 16 }, ...retained,
  ] }, { ...unsupported, readTensorData: async () => { reads++; return new ArrayBuffer(16); },
    writeShard: async () => { writes++; return pin; } }, { modelId: 'preflight-test', converterConfig: createConverterConfig() }),
  /cannot retain.*without copying.*source directory inside/);
  assert.equal(reads, 0); assert.equal(writes, 0); assert.equal(linkCalls, 1);
  assert.deepEqual(await fs.readdir(outputDir), []);
  await assert.rejects(convertModel({ name: 'preflight-test', tensors: retained },
    { readTensorData: async () => { reads++; }, writeShard: async () => { writes++; } },
    { modelId: 'preflight-test', converterConfig: createConverterConfig() }), /cannot retain source files/);
  assert.equal(reads, 0); assert.equal(writes, 0);

  const copying = createNodeConvertIO(outputDir, { ...options, retainedSourceFiles: 'copy' });
  await copying.preflightRetainedSources([sourcePath]);
  const originalOpen = fs.open;
  let partialWrites = 0;
  fs.open = async (...args) => {
    const handle = await originalOpen(...args);
    return args[1] === 'wx' ? {
      write: (buffer, offset, length) => { partialWrites++; return handle.write(buffer, offset, Math.min(length, 3)); },
      close: () => handle.close(),
    } : handle;
  };
  try { await copying.retainSourceFile(sourcePath, 0, pin); }
  finally { fs.open = originalOpen; }
  assert.ok(partialWrites > 1);
  assert.deepEqual(new Uint8Array(await copying.readShardRange(0, 0, 32)), bytes, 'explicit copies handle short file writes');
  await assert.rejects(copying.retainSourceFile(sourcePath, 1, '0'.repeat(64)), /differs from its pin/);
  await assert.rejects(fs.stat(path.join(outputDir, 'shard_00001.bin')), /ENOENT/);
  assert.deepEqual(new Uint8Array(await fs.readFile(sourcePath)), bytes);

  await assert.rejects(appendRetainedSourceShards(retained, {
    retainSourceFile: async () => ({ size: 32, hash: pin, filename: '../outside' }),
  }, [], {}, { hashAlgorithm: 'sha256', manifestConfig: {} }), /relative artifact filename/);
} finally {
  fs.link = originalLink;
  await reader.closeAll();
  await fs.rm(dir, { recursive: true, force: true });
}
console.log('retained source: no-link layout, pin preservation, bounded runtime reads and early rejection passed');
