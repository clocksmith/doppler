// Read n-gram table rows through a converted Flash-Next RDRR artifact and record
// them for an independent source-side comparison (read-source-ngram-rows.py).
//
// Path under test: manifest tensor map -> loader tensor locations ->
// ngramTablePartsFromTensorLocations -> NgramTableReader -> the artifact storage
// context's loadShardRange. Addresses come from NgramTokenHistory for fixed,
// seeded token sequences. Hash verification is off for the bulk read and
// measured separately on a fresh context for one token's rows.
//
// Usage (from the repository root):
//   node artifacts/qwen38-flash-next-20261013/validate-artifact-ngram-reads.js <artifact-dir> <out.json>
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { buildTensorLocations } from '../../src/loader/shard-resolver.js';
import { resolveNgramEmbeddingContract, resolveNgramEmbeddingSession } from '../../src/config/schema/ngram-embedding.schema.js';
import { NgramTokenHistory } from '../../src/inference/pipelines/text/ngram/ngram-addresses.js';
import { NgramTableReader, ngramTablePartsFromTensorLocations } from '../../src/inference/pipelines/text/ngram/ngram-table.js';

const [artifactDir, outPath] = process.argv.slice(2);
if (!artifactDir || !outPath) throw new Error('usage: validate-artifact-ngram-reads.js <artifact-dir> <out.json>');
const manifestBytes = await fs.readFile(path.join(artifactDir, 'manifest.json'));
const manifest = JSON.parse(manifestBytes.toString('utf8'));
const inference = manifest.inference;
const arch = manifest.architecture;
const contract = resolveNgramEmbeddingContract(inference.ngramEmbedding, {
  hiddenSize: arch.hiddenSize, numLayers: arch.numLayers, vocabSize: arch.vocabSize, layerTypes: inference.layerPattern?.layerTypes,
});
if (!contract) throw new Error('manifest declares no ngramEmbedding contract.');
const session = resolveNgramEmbeddingSession(inference.session?.ngramEmbedding, contract);
const layer = contract.layers[0];

const handles = new Map();
const readRange = async (file, offset, length) => {
  let handle = handles.get(file);
  if (!handle) { handle = await fs.open(file, 'r'); handles.set(file, handle); }
  const bytes = new Uint8Array(length);
  let done = 0;
  while (done < length) {
    const { bytesRead } = await handle.read(bytes, done, length - done, offset + done);
    if (bytesRead === 0) throw new Error(`short read at ${file}:${offset + done}`);
    done += bytesRead;
  }
  return bytes;
};
const makeStorage = (verifyHashes) => createArtifactStorageContext({
  manifest, expectedFormat: 'rdrr', verifyHashes,
  readRange: (p, offset, length) => readRange(path.resolve(artifactDir, p), offset, length ?? 0),
  async *streamRange(p, offset, length, options) {
    const chunk = options?.chunkBytes ?? (8 << 20);
    for (let at = 0; at < length; at += chunk) yield readRange(path.resolve(artifactDir, p), offset + at, Math.min(chunk, length - at));
  },
  readText: async (p) => fs.readFile(path.resolve(artifactDir, p), 'utf8'),
  readBinary: async (p) => (await fs.readFile(path.resolve(artifactDir, p))).buffer,
});

const locations = await buildTensorLocations(manifest, {
  hasCustomLoader: true,
  loadTensorsJson: manifest.tensorsFile ? async () => fs.readFile(path.join(artifactDir, manifest.tensorsFile), 'utf8') : undefined,
});
const tensorPrefix = `model.language_model.layers.${layer.decoderLayerIndex}.ple.ple_embedding.ngram_embedding`;
const parts = ngramTablePartsFromTensorLocations(locations, { tensorPrefix, rows: layer.rows, rowWidth: contract.headDim });
const shardsUsed = [...new Set(parts.map((part) => part.source))].sort((a, b) => Number(a) - Number(b));

// Seeded token sequences: uniform ids over the unigram vocabulary plus boundary tokens.
let state = 0x9e3779b9;
const next = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state; };
const sequences = [0, 1, 2, 3].map((s) => Array.from({ length: 256 }, (_, i) => (
  (i + 1) % (61 + s * 17) === 0 ? contract.boundaryTokenId : next() % contract.unigramVocabSize)));

const timed = async (fn) => { const t = performance.now(); const value = await fn(); return { value, ms: performance.now() - t }; };
const storage = makeStorage(false);
const reader = new NgramTableReader({ rows: layer.rows, rowWidth: contract.headDim, dtype: 'bf16', parts, session,
  readRange: (source, offset, length) => storage.loadShardRange(Number(source), offset, length) });
const addresses = [];
const rowDigest = createHash('sha256');
const batches = [];
for (const tokens of sequences) {
  const history = new NgramTokenHistory(contract, layer);
  for (let at = 0; at < tokens.length; at += 32) {
    const rows = history.append(tokens.slice(at, at + 32));
    const { value, ms } = await timed(() => reader.gather(rows));
    rowDigest.update(value);
    addresses.push(...rows);
    batches.push(ms);
  }
}
const cold = reader.stats();
// Re-read the first sequence: every row should now hit the cache.
const history = new NgramTokenHistory(contract, layer);
const again = await timed(() => reader.gather(history.append(sequences[0])));
const warm = reader.stats();

// First access with hash verification on: one token's rows on a fresh context.
const verifying = makeStorage(true);
const verifyReader = new NgramTableReader({ rows: layer.rows, rowWidth: contract.headDim, dtype: 'bf16', parts, session,
  readRange: (source, offset, length) => verifying.loadShardRange(Number(source), offset, length) });
const firstRows = new NgramTokenHistory(contract, layer).append([sequences[0][0]]);
const firstShards = [...new Set([...firstRows].map((row) => parts.find((p) => row >= p.rowBegin && row < p.rowEnd)?.source))];
const verified = await timed(() => verifyReader.gather(firstRows));
const bytesMatchUnverified = Buffer.compare(Buffer.from(verified.value), Buffer.from(await reader.gather(firstRows))) === 0;
for (const handle of handles.values()) await handle.close();
const sorted = [...batches].sort((a, b) => a - b);
const out = {
  schema: 'doppler.qwen4-exp-artifact-ngram-reads/v1',
  modelId: manifest.modelId,
  manifestSha256: createHash('sha256').update(manifestBytes).digest('hex'),
  artifactRoot: manifest.artifactIdentity?.artifactRoot ?? null,
  retainedSourceShards: manifest.conversion?.retainedSourceShards?.length ?? null,
  parts: parts.length, rows: layer.rows, shardsUsed: shardsUsed.length,
  sequences: sequences.length, tokensPerSequence: 256, rowsPerToken: contract.headCount,
  addresses: addresses.length, rowsSha256: rowDigest.digest('hex'),
  addressList: addresses,
  bulk: { verifyHashes: false, batchTokens: 32, batchMs: { p50: sorted[Math.floor(sorted.length / 2)], max: sorted.at(-1) }, stats: cold },
  warmReread: { ms: again.ms, hitsAdded: warm.hits - cold.hits, readsAdded: warm.reads - cold.reads },
  firstAccessVerified: { verifyHashes: true, rows: firstRows.length, distinctShards: firstShards.length, ms: verified.ms,
    bytesMatchUnverified },
};
await fs.writeFile(outPath, JSON.stringify(out, null, 1) + '\n');
console.log(JSON.stringify({ ...out, addressList: `${addresses.length} addresses` }, null, 1));
