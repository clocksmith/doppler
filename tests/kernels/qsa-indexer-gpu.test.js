import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runMatmul } from '../../src/gpu/kernels/matmul.js';
import { runRoPEPrecompute } from '../../src/gpu/kernels/rope-precompute.js';
import { QsaIndexerCache } from '../../src/inference/pipelines/text/indexed-attention/qsa-cache.js';
import { resolveQwen4ExpSourceContracts } from '../../src/converter/qwen4-exp-source.js';
import { resolveIndexedAttentionContract } from '../../src/config/schema/hyper-connection-decoder.schema.js';

// Contract: the QSA indexer operators, driven through the owning cache in the
// upstream chunk schedule, reproduce float64 truth within the derived F32 bound
// (A + B * eps, eps = measured RoPE table error) and select the same blocks as
// the float64 truth and the unchanged upstream indexer wherever the bound
// decides membership. Synthetic dyadic weights; not learned-weight evidence.
const foldedScales = process.argv.includes('--folded-scales');
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical QSA indexer check unavailable: ${probe.reason}`);
const capabilities = getKernelCapabilities();
assert.ok(!/swiftshader|llvmpipe|lavapipe/.test(JSON.stringify(capabilities?.adapterInfo ?? {}).toLowerCase()));
const prep = '../../artifacts/qwen38-flash-next-20261013/';
const readJson = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const prior = await readJson(`${prep}sparse-attention-reference.json`);
const stage = await readJson(`${prep}qsa-stage-reference.json`);
const flashNext = await readJson(`${prep}source/config.json`);
const g = stage.geometry;
const contract = resolveIndexedAttentionContract(resolveQwen4ExpSourceContracts(flashNext).indexedAttention,
  { numLayers: 48, layerTypes: flashNext.text_config.layer_types, attentionRotaryDim: 64 });
assert.equal(contract.denseVisibleLimit, g.denseVisibleLimit);

// Recipe of sparse-attention-reference.json: k * 2^(exponent - 23), k from splitmix64.
const MASK = (1n << 64n) - 1n;
const splitmix = (z) => {
  z = (z + 0x9E3779B97F4A7C15n) & MASK;
  z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK;
  z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK;
  return z ^ (z >> 31n);
};
const dyadic = (count, stream, exponent) => {
  const out = new Float32Array(count);
  const base = BigInt(stream) << 40n;
  const scale = 2 ** (exponent - 23);
  for (let i = 0; i < count; i++) out[i] = (Number(splitmix(base + BigInt(i)) >> 40n) - 2 ** 23) * scale;
  return out;
};
const recipe = Object.fromEntries(prior.recipe.weights.map((w) => [w.name, w]));
const weight = (name) => dyadic(recipe[name].shape.reduce((a, b) => a * b, 1), recipe[name].stream, recipe[name].exponent);
const T = g.sequence;
const H = g.hiddenSize;
const width = (g.queryHeads + 1) * g.headDim;
const hidden = dyadic(T * H, prior.recipe.hidden.stream, prior.recipe.hidden.exponent);
const wqk = weight('indexer.index_qk_proj.weight');
const qNorm = weight('indexer.q_layernorm.weight');
const kNorm = weight('indexer.k_layernorm.weight');

const owned = new Set();
const upload = (data, shape) => {
  const buffer = acquireBuffer(data.byteLength);
  owned.add(buffer);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape);
};
const readF32 = async (buffer, offset, count) => new Float32Array(await readBuffer(buffer, (offset + count) * 4)).slice(offset, offset + count);
const readU32 = async (buffer, offset, count) => new Uint32Array(await readBuffer(buffer, (offset + count) * 4)).slice(offset, offset + count);
const results = {};
let cache = null;
try {
  const rope = await runRoPEPrecompute({ theta: 10000000, rotaryDim: g.rotaryDim, frequencyBaseDim: g.rotaryDim,
    maxSeqLen: 262144, positionCount: T, ropeScale: 1, inverseFrequencies: stage.invFreqF32 });
  // Text-only MRoPE (equal T/H/W positions) equals plain RoPE. The source's f32 inverse frequencies are
  // retained (inference.rope.ropeInverseFrequencies); theta-generated GPU frequencies drift with position.
  owned.add(rope.cos);
  owned.add(rope.sin);
  const half = g.rotaryDim / 2;
  const cosTable = await readF32(rope.cos, 0, T * half);
  const sinTable = await readF32(rope.sin, 0, T * half);
  let eps = 0;
  for (let p = 0; p < T; p++) {
    for (let i = 0; i < half; i++) {
      const angle = Math.fround(p * stage.invFreqF32[i]);
      eps = Math.max(eps, Math.abs(cosTable[p * half + i] - Math.cos(angle)), Math.abs(sinTable[p * half + i] - Math.sin(angle)));
    }
  }
  results.ropeTableError = eps;
  const bound = (entry, index) => entry.boundA[index] + entry.boundB[index] * eps;

  cache = new QsaIndexerCache({ contract, capacity: T });
  const qNormTensor = upload(foldedScales ? Float32Array.from(qNorm, value => 1 + value) : qNorm, [g.headDim]);
  const kNormTensor = upload(foldedScales ? Float32Array.from(kNorm, value => 1 + value) : kNorm, [g.headDim]);
  const wqkBuffer = upload(wqk, [width, H]).buffer;
  const samples = new Map(stage.samples.map((s) => [s.query, s]));
  const querySamples = stage.querySamples;
  const gpu = { selection: new Map(), scores: new Map(), queries: new Map() };
  let start = 0;
  for (const size of stage.chunks) {
    const input = upload(hidden.subarray(start * H, (start + size) * H), [size, H]);
    const projection = await runMatmul(input, wqkBuffer, size, width, H, { transposeB: true, outputDtype: 'f32', bDtype: 'f32' });
    owned.add(projection.buffer);
    const query = await cache.append(null, g.queryHeads >= 1 ? contract.layers[0] : -1,
      { projection, normWeightOffset: !foldedScales, qNormWeight: qNormTensor, kNormWeight: kNormTensor, rope, startPos: start, numTokens: size });
    owned.add(query.buffer);
    const call = await cache.select(null, contract.layers[0], query, { startPos: start, numTokens: size });
    owned.add(call.selection);
    owned.add(call.meta);
    if (call.scores) owned.add(call.scores.buffer);
    for (let i = 0; i < size; i++) {
      const t = start + i;
      if (samples.has(t)) {
        const [count, mode] = await readU32(call.meta, i * 2, 2);
        const list = mode === 1 ? [...await readU32(call.selection, i * call.maxSelected, count)] : null;
        gpu.selection.set(t, { count, mode, list });
        if (mode === 1) gpu.scores.set(t, await readF32(call.scores.buffer, i * call.scores.shape[1], samples.get(t).completeBlocks));
      }
      for (const sample of querySamples.filter((q) => q.token === t)) {
        gpu.queries.set(`${t}:${sample.head}`, await readF32(query.buffer, (i * g.queryHeads + sample.head) * g.headDim, g.headDim));
      }
    }
    start += size;
  }
  assert.equal(cache.length(contract.layers[0]), T);

  // Stage values within A + B * eps.
  const ratio = (actual, entry) => {
    let worst = 0;
    entry.values.forEach((value, index) => { worst = Math.max(worst, Math.abs(actual[index] - value) / bound(entry, index)); });
    return worst;
  };
  results.queryRatio = Math.max(...querySamples.map((s) => ratio(gpu.queries.get(`${s.token}:${s.head}`), s)));
  const { blockKeys } = cache.buffers(contract.layers[0]);
  results.blockKeyRatio = 0;
  for (const sample of stage.blockKeySamples) {
    const actual = await readF32(blockKeys, sample.block * g.headDim, g.headDim);
    results.blockKeyRatio = Math.max(results.blockKeyRatio, ratio(actual, sample));
  }
  assert.ok(results.queryRatio <= 1, `query error/bound ${results.queryRatio}`);
  assert.ok(results.blockKeyRatio <= 1, `block key error/bound ${results.blockKeyRatio}`);

  // Selections: dense shortcut exact; sparse membership decided by the bound.
  results.scoreRatio = 0;
  results.sparse = [];
  const membership = (scores, bounds) => {
    const lower = scores.map((s, j) => s - bounds[j]);
    const upper = scores.map((s, j) => s + bounds[j]);
    const sortedUpper = [...upper].sort((a, b) => b - a);
    const sortedLower = [...lower].sort((a, b) => b - a);
    return scores.map((_, j) => {
      // Guaranteed in: more than blockTopK others cannot all exceed it; out: blockTopK others surely exceed it.
      const othersAbove = sortedUpper.filter((u) => u > lower[j]).length - (upper[j] > lower[j] ? 1 : 0);
      if (othersAbove < g.blockTopK) return 'in';
      const surelyAbove = sortedLower.filter((l) => l > upper[j]).length;
      return surelyAbove >= g.blockTopK ? 'out' : 'ambiguous';
    });
  };
  for (const sample of stage.samples) {
    const actual = gpu.selection.get(sample.query);
    if (sample.kind === 'dense') {
      assert.deepEqual([actual.count, actual.mode], [sample.visible, 0], `dense query ${sample.query}`);
      continue;
    }
    const tail = sample.visible - sample.completeBlocks * g.compressRatio;
    assert.equal(actual.mode, 1);
    assert.equal(actual.count, g.blockTopK * g.compressRatio + tail);
    assert.ok(actual.list.every((v, i) => i === 0 || v > actual.list[i - 1]), 'selection ascending');
    const bounds = sample.scores.map((_, j) => bound(sample, j));
    const gpuScores = gpu.scores.get(sample.query);
    sample.scores.forEach((s, j) => { results.scoreRatio = Math.max(results.scoreRatio, Math.abs(gpuScores[j] - s) / bounds[j]); });
    const toBlocks = (list) => {
      const blocks = new Set();
      for (const token of list) if (token < sample.completeBlocks * g.compressRatio) blocks.add(Math.floor(token / g.compressRatio));
      return blocks;
    };
    const gpuBlocks = toBlocks(actual.list);
    const upstreamBlocks = toBlocks(sample.upstreamSelected);
    assert.equal(gpuBlocks.size, g.blockTopK);
    assert.equal(upstreamBlocks.size, g.blockTopK);
    for (let token = sample.completeBlocks * g.compressRatio; token < sample.visible; token++) {
      assert.ok(actual.list.includes(token) && sample.upstreamSelected.includes(token), 'tail always selected');
    }
    // The selection operator is exact on its own scores: top blockTopK by score, lower block index first on ties.
    const canonical = (value) => (value === 0 ? 0 : value);
    const expectedBlocks = [...gpuScores.keys()]
      .sort((a, b) => canonical(gpuScores[b]) - canonical(gpuScores[a]) || a - b)
      .slice(0, g.blockTopK).sort((a, b) => a - b);
    assert.deepEqual([...gpuBlocks].sort((a, b) => a - b), expectedBlocks, `query ${sample.query} selection on GPU scores`);
    const classes = membership(sample.scores, bounds);
    let ambiguous = 0;
    let agreeUpstream = 0;
    classes.forEach((kind, j) => {
      if (kind === 'in') assert.ok(gpuBlocks.has(j) && upstreamBlocks.has(j), `query ${sample.query} block ${j} must be selected`);
      if (kind === 'out') assert.ok(!gpuBlocks.has(j) && !upstreamBlocks.has(j), `query ${sample.query} block ${j} must be excluded`);
      if (kind === 'ambiguous') ambiguous++;
      if (gpuBlocks.has(j) === upstreamBlocks.has(j)) agreeUpstream++;
      // Disagreement with upstream is admitted only among blocks tied at the boundary score.
      else assert.equal(sample.scores[j], sample.kthScore, `query ${sample.query} block ${j} differs from upstream off a tie`);
    });
    results.sparse.push({ query: sample.query, ambiguousBlocks: ambiguous, agreeWithUpstream: agreeUpstream, blocks: classes.length,
      boundaryTie: sample.kthScore === sample.nextScore, kthScore: sample.kthScore });
  }
  assert.ok(results.scoreRatio <= 1, `score error/bound ${results.scoreRatio}`);

  // Cache ownership contracts.
  const layer = contract.layers[0];
  const probeInput = upload(new Float32Array(width), [1, width]);
  await assert.rejects(cache.append(null, layer, { projection: probeInput, normWeightOffset: !foldedScales, qNormWeight: qNormTensor, kNormWeight: kNormTensor,
    rope, startPos: T - 1, numTokens: 1 }), /contiguous/);
  await assert.rejects(cache.append(null, 5, { projection: probeInput, normWeightOffset: !foldedScales, qNormWeight: qNormTensor, kNormWeight: kNormTensor,
    rope, startPos: 0, numTokens: 1 }), /not an indexed layer/);
  const checkpoint = cache.checkpoint();
  cache.truncate(T - 10);
  assert.equal(cache.length(layer), T - 10);
  assert.throws(() => cache.restore(checkpoint), /predates a truncate/);
  const shorter = cache.checkpoint();
  cache.restore(shorter);
  assert.equal(cache.length(layer), T - 10);
  cache.reset();
  assert.equal(cache.length(layer), 0);
  await assert.rejects(cache.select(null, layer, probeInput, { startPos: 0, numTokens: 1 }), /append first/);
} finally {
  await cache?.destroy();
  await getDevice()?.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
console.log(JSON.stringify({ test: 'qsa-indexer-gpu', adapter: capabilities?.adapterInfo ?? null, results,
  scope: 'Physical QSA indexer operators via the owning cache; synthetic weights; no learned weights or timing' }));
