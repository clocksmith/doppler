import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { applySourceTensorRules } from '../../src/converter/source-tensor-rules.js';
import { appendRetainedSourceShards, partitionRetainedSourceTensors } from '../../src/converter/retained-source-shards.js';
import { createFileRangeReader, createNodeConvertIO, normalizeExecutionConfig } from '../../src/tooling/node-converter/output.js';
import { DEFAULT_CONVERTER_EXECUTION_CONFIG } from '../../src/config/schema/converter.schema.js';
import { computeHash } from '../../src/storage/shards/integrity.js';
import { ngramTablePartsFromTensorLocations } from '../../src/inference/pipelines/text/ngram/ngram-table.js';

// Contract: a `retain` source-tensor rule keeps matched tensors in their pinned
// source files. Conversion links each file into the artifact as one shard after
// the packed shards (no copy), proves its SHA-256 equals the pin, and records
// each tensor's exact byte span; the runtime reads table rows from those spans
// through loadShardRange.
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-retained-'));
const sourceDir = path.join(dir, 'source');
const outputDir = path.join(dir, 'out');
await fs.mkdir(sourceDir);
await fs.mkdir(outputDir);
const fileA = new Uint8Array(96).map((_, i) => i);
const fileB = new Uint8Array(64).map((_, i) => 255 - i);
await fs.writeFile(path.join(sourceDir, 'model-00002.safetensors'), fileA);
await fs.writeFile(path.join(sourceDir, 'model-00001.safetensors'), fileB);
const prefix = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding';
const tensors = [
  { name: `${prefix}.shard_0.weight`, shape: [4, 4], dtype: 'BF16', offset: 16, size: 32, sourcePath: path.join(sourceDir, 'model-00001.safetensors') },
  { name: `${prefix}.shard_1.weight`, shape: [4, 4], dtype: 'BF16', offset: 8, size: 32, sourcePath: path.join(sourceDir, 'model-00002.safetensors') },
  { name: `${prefix}.shard_2.weight`, shape: [2, 4], dtype: 'BF16', offset: 80, size: 16, sourcePath: path.join(sourceDir, 'model-00002.safetensors') },
  { name: 'model.language_model.layers.1.ple.key_proj.weight', shape: [2, 2], dtype: 'BF16', offset: 0, size: 8, sourcePath: path.join(sourceDir, 'model-00002.safetensors') },
];
const files = { 'model-00001.safetensors': sha(fileB), 'model-00002.safetensors': sha(fileA) };
const rule = { kind: 'retain', match: '\\.ngram_embedding\\.shard_\\d+\\.weight$', expectedMatches: 3,
  role: 'ngram_table', group: 'ngram_table', reason: 'n-gram table rows are read in place', files };
const policy = { requireAll: false, rules: [rule] };

const ruled = applySourceTensorRules(tensors, policy);
assert.equal(ruled.length, 4);
assert.deepEqual(ruled[0].retainedSource, { file: 'model-00001.safetensors', sha256: files['model-00001.safetensors'] });
assert.equal(ruled[1].role, 'ngram_table');
assert.equal(ruled[1].group, 'ngram_table');
assert.equal(ruled[3].retainedSource, undefined);
await assert.rejects(async () => applySourceTensorRules(tensors, { requireAll: false, rules: [{ ...rule, files: { 'model-00002.safetensors': files['model-00002.safetensors'] } }] }),
  /lacks a SHA-256 pin for "model-00001.safetensors"/);
await assert.rejects(async () => applySourceTensorRules(tensors, { requireAll: false, rules: [{ ...rule, files: { ...files, 'model-00009.safetensors': files['model-00001.safetensors'] } }] }),
  /pins model-00009.safetensors but no matched tensor is stored there/);
await assert.rejects(async () => applySourceTensorRules(tensors, { requireAll: false, rules: [{ ...rule, group: undefined }] }), /group must be a non-empty string/);

const { packed, retained } = partitionRetainedSourceTensors(ruled);
assert.deepEqual(packed.map((t) => t.name), ['model.language_model.layers.1.ple.key_proj.weight']);
assert.equal(retained.length, 3);

const reader = createFileRangeReader();
const io = createNodeConvertIO(outputDir, { hashAlgorithm: 'sha256', computeHash, readRange: reader.readRange });
const shards = [{ index: 0, filename: 'shard_00000.bin', size: 40, hash: 'packed', offset: 0 }];
const tensorLocations = { packed: { shard: 0, offset: 0, size: 8 } };
const manifestConfig = { hashAlgorithm: 'sha256', conversion: { convertedAt: null } };
const receipt = await appendRetainedSourceShards(retained, io, shards, tensorLocations,
  { hashAlgorithm: 'sha256', manifestConfig });

// Shards follow file-name order after the packed shard, with cumulative offsets.
assert.deepEqual(shards.slice(1), [
  { index: 1, filename: 'shard_00001.bin', size: 64, hash: files['model-00001.safetensors'], offset: 40 },
  { index: 2, filename: 'shard_00002.bin', size: 96, hash: files['model-00002.safetensors'], offset: 104 },
]);
for (const shard of shards.slice(1)) {
  const linked = await fs.stat(path.join(outputDir, shard.filename));
  const source = await fs.stat(path.join(sourceDir, receipt.find((entry) => entry.shard === shard.index).file));
  assert.equal(linked.ino, source.ino, 'the shard is a link to the source file, not a copy');
}
assert.deepEqual(tensorLocations[`${prefix}.shard_2.weight`],
  { shard: 2, offset: 80, size: 16, shape: [2, 4], dtype: 'BF16', role: 'ngram_table', group: 'ngram_table' });
assert.deepEqual(manifestConfig.conversion, { convertedAt: null, retainedSourceShards: receipt });
assert.deepEqual(receipt.map((entry) => [entry.shard, entry.file, entry.bytes, entry.tensors]),
  [[1, 'model-00001.safetensors', 64, 1], [2, 'model-00002.safetensors', 96, 2]]);
const span = new Uint8Array(await io.readShardRange(2, 80, 16));
assert.deepEqual([...span], [...fileA.subarray(80, 96)], 'tensor spans address the retained bytes');

// A pin mismatch removes the link and fails; other hashes and surfaces fail closed.
const wrong = retained.map((t) => ({ ...t, retainedSource: { ...t.retainedSource, sha256: '0'.repeat(64) } }));
await assert.rejects(appendRetainedSourceShards(wrong.slice(0, 1), io, [], {}, { hashAlgorithm: 'sha256', manifestConfig: {} }),
  /differs from its pin/);
await assert.rejects(fs.stat(path.join(outputDir, 'shard_00000.bin')), /ENOENT/);
await assert.rejects(appendRetainedSourceShards(retained, io, [], {}, { hashAlgorithm: 'blake3', manifestConfig: {} }),
  /require manifest.hashAlgorithm "sha256"/);
await assert.rejects(appendRetainedSourceShards(retained, {}, [], {}, { hashAlgorithm: 'sha256', manifestConfig: {} }),
  /cannot link source files/);
const outOfRange = [{ ...retained[0], offset: 60 }];
await assert.rejects(appendRetainedSourceShards(outOfRange, io, [], {}, { hashAlgorithm: 'sha256', manifestConfig: {} }),
  /lies outside model-00001.safetensors/);
assert.deepEqual(await appendRetainedSourceShards([], io, shards, tensorLocations, { hashAlgorithm: 'blake3' }), []);

// An explicit copy writes the same bytes to a new file; the shard hash is still the pin.
const copyDir = path.join(dir, 'copy');
await fs.mkdir(copyDir);
const copyIo = createNodeConvertIO(copyDir, { hashAlgorithm: 'sha256', computeHash, readRange: reader.readRange, retainedSourceFiles: 'copy' });
const copied = await copyIo.retainSourceFile(path.join(sourceDir, 'model-00002.safetensors'), 3, files['model-00002.safetensors']);
assert.deepEqual(copied, { size: 96, hash: files['model-00002.safetensors'] });
const copyStat = await fs.stat(path.join(copyDir, 'shard_00003.bin'));
assert.notEqual(copyStat.ino, (await fs.stat(path.join(sourceDir, 'model-00002.safetensors'))).ino, 'copy mode does not link');
assert.deepEqual([...await fs.readFile(path.join(copyDir, 'shard_00003.bin'))], [...fileA]);
await assert.rejects(copyIo.retainSourceFile(path.join(sourceDir, 'model-00001.safetensors'), 4, '0'.repeat(64)), /differs from its pin/);
await assert.rejects(fs.stat(path.join(copyDir, 'shard_00004.bin')), /ENOENT/);
assert.throws(() => normalizeExecutionConfig({ retainedSourceFiles: 'move' }, DEFAULT_CONVERTER_EXECUTION_CONFIG), /must be "link" or "copy"/);
assert.equal(normalizeExecutionConfig(null, DEFAULT_CONVERTER_EXECUTION_CONFIG).retainedSourceFiles, 'link');
await reader.closeAll();

// Runtime: the loader's normalized locations become table parts addressed by shard index.
const normalized = new Map(Object.entries(tensorLocations).filter(([name]) => name !== 'packed')
  .map(([name, location]) => [name, { ...location, shardIndex: location.shard, spans: undefined }]));
normalized.set('model.language_model.layers.1.ple.key_proj.weight', { shardIndex: 0, offset: 0, size: 8, shape: [2, 2], dtype: 'F16', role: 'matmul' });
const parts = ngramTablePartsFromTensorLocations(normalized, { tensorPrefix: prefix, rows: 10, rowWidth: 4 });
assert.deepEqual(parts, [
  { rowBegin: 0, rowEnd: 4, source: 1, byteOffset: 16, byteLength: 32 },
  { rowBegin: 4, rowEnd: 8, source: 2, byteOffset: 8, byteLength: 32 },
  { rowBegin: 8, rowEnd: 10, source: 2, byteOffset: 80, byteLength: 16 },
]);
assert.throws(() => ngramTablePartsFromTensorLocations(normalized, { tensorPrefix: prefix, rows: 12, rowWidth: 4 }), /cover 10 rows; the contract declares 12/);
const gap = new Map(normalized);
gap.set(`${prefix}.shard_4.weight`, gap.get(`${prefix}.shard_2.weight`));
assert.throws(() => ngramTablePartsFromTensorLocations(gap, { tensorPrefix: prefix, rows: 10, rowWidth: 4 }), /shard_4.weight is not part of the contiguous/);
const split = new Map(normalized);
split.set(`${prefix}.shard_1.weight`, { ...split.get(`${prefix}.shard_1.weight`), spans: [{}, {}] });
assert.throws(() => ngramTablePartsFromTensorLocations(split, { tensorPrefix: prefix, rows: 10, rowWidth: 4 }), /must lie in one shard/);
const transcoded = new Map(normalized);
transcoded.set(`${prefix}.shard_0.weight`, { ...transcoded.get(`${prefix}.shard_0.weight`), dtype: 'F16' });
assert.throws(() => ngramTablePartsFromTensorLocations(transcoded, { tensorPrefix: prefix, rows: 10, rowWidth: 4 }), /retained BF16 ngram_table/);

await fs.rm(dir, { recursive: true, force: true });
console.log('retained-source-shards.test: ok');
