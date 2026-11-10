#!/usr/bin/env node
// Diagnostic only: a common-weights RDRR whose GGUF tensors that Pooled requantizes at load
// (Q4_1, Q5_0, Q5_K, Q6_K) hold Pooled's prepared Q8 values, laid out as Q8_0 blocks (the
// f16 scale and the 32 int8 values Pooled stores, unchanged: no second quantization step).
// Every other tensor keeps the source RDRR's bytes. The values come from Pooled's own
// requantQ8Streaming in the supplied, read-only vendor tree. Not a product artifact and not
// native-weight evidence; the output directory must be new.
//   node tools/experiments/build-pooled-q8-common-rdrr.js --source-rdrr <dir> --gguf <file> \
//     --gguf-sha256 <hex> --pooled-root <dir> --out <new dir>
// Admission: the GGUF must hash to --gguf-sha256, the source RDRR must name that GGUF as its
// source, and every source RDRR tensor (replaced or kept) must hold exactly that GGUF tensor's
// bytes and element count, so the kept and replaced tensors come from one checkpoint. The
// vendor files whose code produces the replacement values are hashed into the inventory.
// Then: node tools/refresh-converted-manifest.js <out> --config <conversion config> --model-id <id>
// (shard BLAKE3 hashes, block Merkle roots), and finally
//   node tools/experiments/build-pooled-q8-common-rdrr.js --finalize <out>
// which restores this artifact's quantizationInfo (the refresh re-derives it from the
// conversion config) and recomputes artifactIdentity from the new shard set.
import { createHash } from 'node:crypto';
import { copyFile, mkdir, open, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { portableSourcePath } from '../../src/tooling/portable-source-path.js';

const REQUANTIZED = new Set(['Q4_1', 'Q5_0', 'Q5_K', 'Q6_K']);
const SHARD_BYTES = 64 * 1024 * 1024;
const Q8_BLOCK_BYTES = 34;

const { values: args } = parseArgs({ options: {
  'source-rdrr': { type: 'string' }, gguf: { type: 'string' }, 'gguf-sha256': { type: 'string' },
  'pooled-root': { type: 'string' }, out: { type: 'string' }, finalize: { type: 'string' },
} });
if (args.finalize) {
  const { buildArtifactIdentity } = await import('../../src/converter/artifact-identity.js');
  const manifestPath = path.join(args.finalize, 'manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const intended = manifest.metadata?.diagnosticCommonWeights?.quantizationInfo;
  if (!intended) throw new Error('Not a common-weights diagnostic manifest.');
  manifest.quantizationInfo = intended;
  const { shardSetHash, weightCapsuleHash, weightCapsuleId, manifestVariantId, ...kept } = manifest.artifactIdentity;
  manifest.artifactIdentity = await buildArtifactIdentity({
    explicitArtifactIdentity: kept,
    hashString: async (text) => `sha256:${createHash('sha256').update(text).digest('hex')}`,
    hashAlgorithm: manifest.hashAlgorithm, shards: manifest.shards, modelType: manifest.modelType,
    quantizationInfo: manifest.quantizationInfo, inference: manifest.inference, modelId: manifest.modelId,
  });
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`finalized ${manifest.modelId}: shardSetHash ${manifest.artifactIdentity.shardSetHash}, weightCapsuleId ${manifest.artifactIdentity.weightCapsuleId}`);
  process.exit(0);
}
for (const key of ['source-rdrr', 'gguf', 'gguf-sha256', 'pooled-root', 'out']) {
  if (!args[key]) throw new Error(`--${key} is required.`);
}
if (await stat(args.out).then(() => true, () => false)) throw new Error(`Output ${args.out} already exists.`);

const gguf = await import(pathToFileURL(path.join(args['pooled-root'], 'engine/gguf.js')).href);
const source = JSON.parse(await readFile(path.join(args['source-rdrr'], 'manifest.json'), 'utf8'));
const file = await open(args.gguf, 'r');
const read = async (offset, length) => {
  const bytes = new Uint8Array(length);
  const { bytesRead } = await file.read(bytes, 0, length, offset);
  if (bytesRead !== length) throw new Error(`Short GGUF read at ${offset}.`);
  return bytes;
};
const header = gguf.parseGGUFHeader((await read(0, 64 << 20)).buffer);
const typeName = { [gguf.GGML_Q4_1]: 'Q4_1', [gguf.GGML_Q5_0]: 'Q5_0', [gguf.GGML_Q5_K]: 'Q5_K', [gguf.GGML_Q6_K]: 'Q6_K' };
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// Admission: one checkpoint, one vendor tree.
const ggufHash = createHash('sha256');
for (let offset = 0, size = (await file.stat()).size; offset < size; offset += 64 << 20) {
  ggufHash.update(await read(offset, Math.min(64 << 20, size - offset)));
}
const ggufSha256 = ggufHash.digest('hex');
if (ggufSha256 !== args['gguf-sha256']) throw new Error(`GGUF sha256 ${ggufSha256} is not the expected ${args['gguf-sha256']}.`);
if (![portableSourcePath(args.gguf), pathToFileURL(path.resolve(args.gguf)).href].includes(source.metadata?.source)) {
  throw new Error(`Source RDRR names ${source.metadata?.source}, not ${args.gguf}, as its source.`);
}
/** @param {{ shard?: number, offset?: number, size: number, spans?: { shardIndex: number, offset: number, size: number }[] }} entry */
const readSourceTensor = async (entry) => {
  const spans = entry.spans ?? [{ shardIndex: entry.shard, offset: entry.offset, size: entry.size }];
  const bytes = new Uint8Array(entry.size);
  let at = 0;
  for (const span of spans) {
    const shard = await open(path.join(args['source-rdrr'], source.shards[span.shardIndex].filename), 'r');
    await shard.read(bytes, at, span.size, span.offset);
    await shard.close();
    at += span.size;
  }
  return bytes;
};
for (const [name, entry] of Object.entries(source.tensors)) {
  const info = header.tensors[name];
  const elements = entry.shape.reduce((a, b) => a * b, 1);
  if (!info || info.nElems !== elements || info.byteLength !== entry.size) throw new Error(`${name}: not the GGUF tensor of the same extent.`);
  if (sha256(await readSourceTensor(entry)) !== sha256(await read(info.byteOffset, info.byteLength))) {
    throw new Error(`${name}: source RDRR bytes differ from the GGUF tensor.`);
  }
}
const vendorFiles = (await readdir(path.join(args['pooled-root'], 'engine'), { recursive: true }))
  .filter((name) => name.endsWith('.js')).sort();
const vendorTree = createHash('sha256');
for (const name of vendorFiles) vendorTree.update(`${name}\0${sha256(await readFile(path.join(args['pooled-root'], 'engine', name)))}\n`);
const admission = {
  ggufSha256: `sha256:${ggufSha256}`,
  sourceTensorsMatchingGguf: Object.keys(source.tensors).length,
  pooledGgufJs: `sha256:${sha256(await readFile(path.join(args['pooled-root'], 'engine/gguf.js')))}`,
  pooledEngineTree: `sha256:${vendorTree.digest('hex')}`,
  pooledEngineFiles: vendorFiles.length,
  builder: `sha256:${sha256(await readFile(new URL(import.meta.url)))}`,
};
console.log(`admission: ${JSON.stringify(admission)}`);

await mkdir(args.out);
// Every source file but the manifest (shards, tokenizer, origin records) is copied as is.
for (const name of await readdir(args['source-rdrr'])) {
  if (name !== 'manifest.json') await copyFile(path.join(args['source-rdrr'], name), path.join(args.out, name));
}

const manifest = structuredClone(source);
const shards = manifest.shards;
let current = new Uint8Array(SHARD_BYTES), used = 0;
const flush = async () => {
  if (used === 0) return;
  const index = shards.length;
  const bytes = current.subarray(0, used);
  const filename = `shard_${String(index).padStart(5, '0')}.bin`;
  await writeFile(path.join(args.out, filename), bytes);
  const previous = shards[index - 1];
  shards.push({ index, filename, size: used, hash: sha256(bytes), offset: previous.offset + previous.size });
  current = new Uint8Array(SHARD_BYTES); used = 0;
};
/** @param {Uint8Array} bytes */
const append = async (bytes) => {
  const spans = [];
  for (let at = 0; at < bytes.length;) {
    if (used === SHARD_BYTES) await flush();
    const take = Math.min(bytes.length - at, SHARD_BYTES - used);
    current.set(bytes.subarray(at, at + take), used);
    spans.push({ shardIndex: shards.length, offset: used, size: take });
    used += take; at += take;
  }
  return spans;
};

const replaced = [];
for (const [name, entry] of Object.entries(manifest.tensors)) {
  if (!REQUANTIZED.has(entry.dtype)) continue;
  const info = header.tensors[name];
  if (!info || typeName[info.ggmlType] !== entry.dtype) throw new Error(`${name}: GGUF type does not match the RDRR dtype ${entry.dtype}.`);
  const cols = entry.shape[entry.shape.length - 1];
  const rows = info.nElems / cols;
  if (!Number.isInteger(rows) || cols % 32) throw new Error(`${name}: rows of whole 32-value blocks required.`);
  const { qs, scales } = gguf.requantQ8Streaming({ ggmlType: info.ggmlType, nElems: info.nElems, shape: [rows, cols] },
    await read(info.byteOffset, info.byteLength));
  const scaleHalves = new Uint8Array(scales.buffer, scales.byteOffset, scales.byteLength);
  const blocks = info.nElems / 32;
  const q8 = new Uint8Array(blocks * Q8_BLOCK_BYTES);
  for (let b = 0; b < blocks; b++) {
    q8[b * Q8_BLOCK_BYTES] = scaleHalves[2 * b];
    q8[b * Q8_BLOCK_BYTES + 1] = scaleHalves[2 * b + 1];
    q8.set(qs.subarray(b * 32, b * 32 + 32), b * Q8_BLOCK_BYTES + 2);
  }
  const spans = await append(q8);
  delete entry.shard; delete entry.offset; delete entry.spans;
  if (spans.length === 1) { entry.shard = spans[0].shardIndex; entry.offset = spans[0].offset; } else entry.spans = spans;
  replaced.push({ name, sourceDtype: entry.dtype, bytes: q8.length,
    pooledQs: `sha256:${sha256(qs.subarray(0, info.nElems))}`, pooledScales: `sha256:${sha256(scaleHalves.subarray(0, 2 * blocks))}`,
    q8_0: `sha256:${sha256(q8)}` });
  entry.dtype = 'Q8_0';
  entry.size = q8.length;
  console.log(`${name}: ${replaced.at(-1).sourceDtype} -> Pooled Q8 as Q8_0 (${q8.length} bytes)`);
}
await flush();
await file.close();

manifest.modelId = `${source.modelId}-pooled-q8-common`;
manifest.totalSize = shards.reduce((sum, shard) => sum + shard.size, 0);
manifest.quantizationInfo = { ...manifest.quantizationInfo, lmHead: 'q8_0', variantTag: `${manifest.quantizationInfo.variantTag}-pooledq8` };
manifest.metadata = { ...manifest.metadata, diagnosticCommonWeights: {
  quantizationInfo: manifest.quantizationInfo,
  scope: 'Diagnostic common-weights lane only: Pooled-prepared Q8 values for the tensors Pooled requantizes; not a product artifact or native-weight result.',
  sourceRdrrModelId: source.modelId, sourceShardSetHash: source.artifactIdentity?.shardSetHash ?? null,
  pooledRoot: path.resolve(args['pooled-root']), gguf: path.resolve(args.gguf), admission, replaced,
} };
await writeFile(path.join(args.out, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(path.join(args.out, 'pooled-q8-common.json'), `${JSON.stringify({ admission, replaced }, null, 2)}\n`);
console.log(`${replaced.length} tensors replaced; ${shards.length - source.shards.length} shards appended; manifest written.`);
