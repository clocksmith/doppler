import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolveQwen4ExpSourceContracts } from '../../src/converter/qwen4-exp-source.js';
import { resolveNgramEmbeddingContract, resolveNgramEmbeddingSession } from '../../src/config/schema/ngram-embedding.schema.js';
import { NgramTokenHistory } from '../../src/inference/pipelines/text/ngram/ngram-addresses.js';
import { NgramTableReader, ngramTablePartsFromCheckpointHeaders } from '../../src/inference/pipelines/text/ngram/ngram-table.js';

// Contract: n-gram addresses equal the pinned upstream addresses exactly,
// independent of chunking and across interleaved histories; history state
// truncates and restores consistently; the table reader maps rows to the
// authenticated checkpoint spans and enforces declared cache, staging,
// concurrency and cancellation bounds.
const prep = '../../artifacts/qwen38-flash-next-20261013/';
const readJson = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const flashNext = await readJson(`${prep}source/config.json`);
const fixtures = await readJson(`${prep}reference-fixtures.json`);
const incremental = await readJson(`${prep}ngram-incremental-reference.json`);
const headers = await readJson(`${prep}checkpoint-headers.json`);
const contract = resolveNgramEmbeddingContract(resolveQwen4ExpSourceContracts(flashNext).ngramEmbedding,
  { hiddenSize: 2560, numLayers: 48, vocabSize: 248320, layerTypes: flashNext.text_config.layer_types });
const [layer] = contract.layers;
const EOS = contract.boundaryTokenId;

// 1. Pinned upstream addresses for 8 tokens with boundaries.
const tokens = fixtures.ngram.tokens[0];
assert.deepEqual(tokens, incremental.tokens);
const expected = fixtures.ngram.addresses.values[0].flat();
const full = new NgramTokenHistory(contract, layer).append(tokens);
assert.deepEqual([...full], expected);

// 2. Every chunk partition reproduces the full-sequence addresses.
for (let mask = 0; mask < 1 << (tokens.length - 1); mask++) {
  const history = new NgramTokenHistory(contract, layer, 1);
  const parts = [];
  let start = 0;
  for (let i = 1; i <= tokens.length; i++) {
    if (i === tokens.length || mask & (1 << (i - 1))) {
      parts.push(...history.append(tokens.slice(start, i)));
      start = i;
    }
  }
  assert.deepEqual(parts, expected, `partition mask ${mask}`);
}

// 3. Independent literal port of the torch operations (cummax segments, gather, where) agrees on
//    boundary-dense random sequences, and interleaved histories do not interact.
function literal(sequence) {
  const context = contract.ngramSize - 1;
  const history = [...Array(context).fill(EOS), ...sequence];
  const positions = history.map((_, i) => i);
  const eos = history.map((t, i) => (t === EOS ? i : -1));
  const inclusive = eos.map((_, i) => Math.max(...eos.slice(0, i + 1)));
  const previous = [-1, ...inclusive.slice(0, -1)];
  const shifted = (shift) => history.map((t, i) => {
    if (shift === 0) return t;
    const valid = positions[i] - (previous[i] + 1) >= shift && i - shift >= 0;
    return valid ? history[Math.max(i - shift, 0)] : EOS;
  });
  const shifts = Array.from({ length: contract.ngramSize }, (_, k) => shifted(k).map(BigInt));
  const out = [];
  for (let i = context; i < history.length; i++) {
    for (let n = 2; n <= contract.ngramSize; n++) {
      let mixed = shifts[0][i] * layer.multipliers[0];
      for (let p = 1; p < n; p++) mixed ^= shifts[p][i] * layer.multipliers[p];
      for (let h = (n - 2) * contract.headsPerNgram; h < (n - 1) * contract.headsPerNgram; h++) {
        out.push(Number(mixed % BigInt(layer.headVocabSizes[h])) + layer.headOffsets[h]);
      }
    }
  }
  return out;
}
let seed = 7;
const random = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 2 ** 32; };
const sequenceOf = (length) => Array.from({ length }, () => (random() < 0.25 ? EOS : Math.floor(random() * contract.unigramVocabSize)));
for (let trial = 0; trial < 20; trial++) {
  const a = sequenceOf(40);
  const b = sequenceOf(40);
  const ha = new NgramTokenHistory(contract, layer);
  const hb = new NgramTokenHistory(contract, layer);
  const outA = [];
  const outB = [];
  for (let i = 0; i < a.length;) {
    const step = 1 + Math.floor(random() * 5);
    outA.push(...ha.append(a.slice(i, i + step)));
    outB.push(...hb.append(b.slice(i, i + step)));
    i += step;
  }
  assert.deepEqual(outA, literal(a));
  assert.deepEqual(outB, literal(b));
}
// Leading boundary tokens, consecutive boundaries and the largest token id stay exact.
const edge = [EOS, EOS, 0, EOS, contract.unigramVocabSize - 1, contract.unigramVocabSize - 1, EOS];
assert.deepEqual([...new NgramTokenHistory(contract, layer).append(edge)], literal(edge));
assert.throws(() => new NgramTokenHistory(contract, layer).append([contract.unigramVocabSize]), /outside/);
assert.throws(() => new NgramTokenHistory(contract, layer).append([1.5]), /outside/);

// 4. History truncation, checkpoint and restore.
{
  const history = new NgramTokenHistory(contract, layer);
  history.append(tokens.slice(0, 5));
  const checkpoint = history.checkpoint();
  history.append(tokens.slice(5));
  history.restore(checkpoint);
  assert.deepEqual([...history.append(tokens.slice(5))], expected.slice(5 * 16));
  history.truncate(3);
  assert.throws(() => history.restore(checkpoint), /not restorable/);
  assert.deepEqual([...history.append(tokens.slice(3))], expected.slice(3 * 16));
  history.reset();
  assert.deepEqual([...history.append(tokens)], expected);
}

// 5. Authenticated row-to-part mapping of the Flash-Next checkpoint.
const parts = ngramTablePartsFromCheckpointHeaders(headers);
assert.equal(parts.length, 128);
const session = (rowCache, stagingRows = 64, maxConcurrent = 4) => resolveNgramEmbeddingSession(
  { rowCache: { ...rowCache, eviction: 'lru' }, staging: { maxRows: stagingRows }, reads: { maxConcurrent } }, contract);
const reads = [];
let inFlight = 0;
let peak = 0;
const fakeRange = async (source, offset, length, signal) => {
  inFlight += 1;
  peak = Math.max(peak, inFlight);
  reads.push({ source, offset, length });
  await new Promise((resolve) => setTimeout(resolve, 1));
  inFlight -= 1;
  if (signal?.aborted) throw signal.reason;
  // Row r's bytes encode r so the order can be checked.
  const out = new Uint8Array(length);
  const view = new DataView(out.buffer);
  const part = parts.find((p) => p.source === source && offset >= p.byteOffset && offset < p.byteOffset + p.byteLength);
  for (let i = 0; i < length / 320; i++) view.setUint32(i * 320, part.rowBegin + (offset - part.byteOffset) / 320 + i, true);
  return out;
};
const table = new NgramTableReader({ rows: layer.rows, rowWidth: contract.headDim, dtype: contract.tableDtype, parts,
  readRange: fakeRange, session: session({ maxRows: 6, maxBytes: 6 * 320 }) });
const shardRows = headers.ngramTable.shardRows[0];
const probe = [0, shardRows - 1, shardRows, shardRows + 1, layer.rows - 1, shardRows];
const gathered = await table.gather(probe);
const rowAt = (bytes, i) => new DataView(bytes.buffer).getUint32(i * 320, true);
assert.deepEqual(probe.map((_, i) => rowAt(gathered, i)), probe);
const lastShard = headers.ngramTable.shards[127];
assert.ok(reads.some((r) => r.source === lastShard.file && r.offset === lastShard.fileByteEnd - 320 && r.length === 320));
// Rows shardRows and shardRows + 1 coalesce into one read; row shardRows - 1 belongs to the previous shard.
assert.ok(reads.some((r) => r.source === headers.ngramTable.shards[1].file && r.length === 640));
assert.equal(table.stats().misses, 5);
assert.equal(table.stats().rowBytesVerified, false);
await table.gather([0, shardRows]);
assert.equal(table.stats().hits, 2);
await assert.rejects(table.gather([layer.rows]), /outside/);

// 6. Bounds: the LRU holds at most maxRows and maxBytes; staging batches; concurrency cap.
{
  reads.length = 0;
  peak = 0;
  const bounded = new NgramTableReader({ rows: layer.rows, rowWidth: contract.headDim, dtype: 'bf16', parts,
    readRange: fakeRange, session: session({ maxRows: 1000, maxBytes: 3 * 320 }, 16, 2) });
  const scattered = Array.from({ length: 40 }, (_, i) => i * 1000003 % layer.rows);
  await bounded.gather(scattered);
  const stats = bounded.stats();
  assert.ok(stats.cachedRows <= 3 && stats.cachedBytes <= 3 * 320, 'byte bound applies');
  assert.ok(peak <= 2, `concurrency ${peak} exceeds the declared bound`);
  assert.ok(reads.length >= Math.ceil(40 / 16), 'staging splits reads into batches');
}

// 7. Cancellation aborts in-flight reads and caches nothing from the aborted batch.
{
  const controller = new AbortController();
  const slow = new NgramTableReader({ rows: layer.rows, rowWidth: contract.headDim, dtype: 'bf16', parts,
    readRange: async (source, offset, length, signal) => {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, 50);
        signal?.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason); }, { once: true });
      });
      return new Uint8Array(length);
    },
    session: session({ maxRows: 100, maxBytes: 100 * 320 }) });
  const pending = slow.gather([1, 2, 10, 20], { signal: controller.signal });
  controller.abort(new Error('cancelled by test'));
  await assert.rejects(pending, /cancelled by test/);
  assert.equal(slow.stats().cachedRows, 0);
  await assert.rejects(slow.gather([1], { signal: controller.signal }), /cancelled by test/);
}

// 8. Malformed part maps fail.
assert.throws(() => new NgramTableReader({ rows: layer.rows, rowWidth: 160, dtype: 'bf16', parts: parts.slice(1),
  readRange: fakeRange, session: session({ maxRows: 1, maxBytes: 320 }) }), /tile/);
assert.throws(() => new NgramTableReader({ rows: layer.rows, rowWidth: 160, dtype: 'f32', parts,
  readRange: fakeRange, session: session({ maxRows: 1, maxBytes: 320 }) }), /unsupported table dtype/);

console.log('ngram-embedding-host.test: ok');
