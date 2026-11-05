import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  assertConvertedQwen4ExpContracts,
  resolveQwen4ExpSourceContracts,
} from '../../src/converter/qwen4-exp-source.js';
import sourceDefaults from '../../src/converter/qwen4-exp-source-defaults.json' with { type: 'json' };
import { resolveHyperConnectionDecoderContracts } from '../../src/config/schema/hyper-connection-decoder.schema.js';

// Contract: the pinned qwen4_exp source config lowers to explicit Doppler
// contracts whose derived values equal the checkpoint's retained buffers, and
// omitted source fields resolve from the pinned config class, not another family.
const readJson = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const prep = '../../artifacts/qwen38-flash-next-20261013/';
const flashNext = await readJson(`${prep}source/config.json`);
const headers = await readJson(`${prep}checkpoint-headers.json`);
const binding = await readJson(`${prep}source-binding.json`);
const configClassSource = await readFile(new URL(`${prep}source/configuration_qwen4_exp.py`, import.meta.url), 'utf8');

// The defaults mirror names the bound configuration class and equals its literals.
const bound = binding.referenceImplementation.files.find((entry) => entry.path === 'configuration_qwen4_exp.py');
assert.equal(sourceDefaults.source.sha256, bound.sha256);
assert.equal(sourceDefaults.source.url, bound.url);
const textClass = configClassSource.slice(
  configClassSource.indexOf('class Qwen4ExpTextConfig'),
  configClassSource.indexOf('class Qwen4ExpVisionConfig'),
);
const pythonLiteral = (text) => {
  if (text === 'None') return null;
  if (text === 'True') return true;
  if (text === 'False') return false;
  return JSON.parse(text.replaceAll('_', ''));
};
for (const [field, value] of Object.entries(sourceDefaults.defaults)) {
  const match = textClass.match(new RegExp(`^    ${field}: [^=\\n]+ = (.+)$`, 'm'));
  assert.ok(match, `${field} has a class-level default`);
  assert.deepEqual(pythonLiteral(match[1].trim()), value, `${field} default`);
}

// Flash-Next omits seed and norm_topk_prob; the class default seed 1234 applies.
assert.equal(flashNext.text_config.seed, undefined);
const contracts = resolveQwen4ExpSourceContracts(flashNext);
assert.deepEqual(contracts.hyperConnections, {
  semantics: 'qwen4_exp_gated_residual', streamCount: 4, streamWidth: 2560, projectionRank: 320, normEps: 1e-6,
});
const qsa = contracts.indexedAttention;
assert.deepEqual(qsa.layers, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
assert.equal(qsa.rotaryDim, 64);
assert.equal(qsa.blockTopK, 512);
assert.equal(qsa.denseVisibleLimit, 2051);
assert.equal(qsa.forceSelfToken, false);

const ple = contracts.ngramEmbedding;
assert.equal(ple.seed, 1234);
assert.equal(ple.boundaryTokenId, 248044);
assert.equal(ple.headDim, 160);
assert.equal(ple.convStateLength, 9);
assert.equal(ple.layers.length, 1);
const [layer] = ple.layers;
// One-based source id [2] selects zero-based decoder index 1.
assert.equal(layer.sourceLayerId, 2);
assert.equal(layer.decoderLayerIndex, 1);
// Every derived value equals the checkpoint buffers read at the pinned revision.
assert.deepEqual(layer.multipliers, headers.buffers.layer_multipliers.values);
assert.deepEqual(layer.headOffsets.map(String), headers.buffers.ngram_heads_offsets.values);
assert.deepEqual(layer.headVocabSizes.map(String), headers.buffers.ngram_heads_vocab_sizes.values);
assert.equal(layer.rows, headers.ngramTable.rows);
assert.equal(layer.unpaddedRows, headers.ngramTable.unpaddedRows);
// The largest multiplier is beyond Number's exact range only through products; it must stay a string.
assert.ok(BigInt(layer.multipliers[0]) * 248319n > BigInt(Number.MAX_SAFE_INTEGER));

// A different seed changes the hash multipliers but not the prime table.
const reseeded = resolveQwen4ExpSourceContracts({ ...flashNext, text_config: { ...flashNext.text_config, seed: 7 } });
assert.notDeepEqual(reseeded.ngramEmbedding.layers[0].multipliers, layer.multipliers);
assert.deepEqual(reseeded.ngramEmbedding.layers[0].headVocabSizes, layer.headVocabSizes);

// The derived contracts validate at runtime against the source geometry.
const geometry = {
  hiddenSize: 2560, numLayers: 48, vocabSize: 248320, layerTypes: flashNext.text_config.layer_types,
  attentionRotaryDim: 64, modelId: 'flash-next',
};
const session = { rowCache: { maxRows: 4096, maxBytes: 4096 * 320, eviction: 'lru' }, staging: { maxRows: 512 }, reads: { maxConcurrent: 8 } };
const inference = {
  ...contracts,
  session: { ngramEmbedding: session },
  normalization: { postAttentionNorm: false, preFeedforwardNorm: false, postFeedforwardNorm: false },
  layerPattern: { residualBranchScale: 1 },
};
const resolved = resolveHyperConnectionDecoderContracts(inference, geometry);
assert.equal(resolved.ngramEmbedding.layers[0].multipliers[0], 23703573157769n);

// Source validation mirrors upstream validate_architecture.
const text = (patch) => ({ ...flashNext, text_config: { ...flashNext.text_config, ...patch } });
assert.throws(() => resolveQwen4ExpSourceContracts(text({ hc_count: 1 })), /hc_count > 1/);
assert.throws(() => resolveQwen4ExpSourceContracts(text({ indexer_kv_heads: 2 })), /indexer_kv_heads=1/);
assert.throws(() => resolveQwen4ExpSourceContracts(text({ indexer_budget: null })), /missing required fields: indexer_budget/);
assert.throws(() => resolveQwen4ExpSourceContracts(text({ indexer_budget: 2050 })), /divisible/);
assert.throws(() => resolveQwen4ExpSourceContracts(text({ ple_layer_ids: [4] })), /only supported on linear_attention/);
assert.throws(() => resolveQwen4ExpSourceContracts(text({ ple_layer_ids: [0] })), /one-indexed/);
assert.throws(() => resolveQwen4ExpSourceContracts(text({ eos_token_id: null })), /eos_token_id/);
assert.throws(() => resolveQwen4ExpSourceContracts(text({ ple_embed_dim: 2561 })), /divisible by 16/);
assert.throws(() => resolveQwen4ExpSourceContracts(text({ dtype: 'float8_e4m3fn' })), /not a supported stored dtype/);
assert.equal(resolveQwen4ExpSourceContracts(text({ ple_layer_ids: [] })).ngramEmbedding, null);
assert.equal(resolveQwen4ExpSourceContracts({ model_type: 'qwen3_5' }), null);
// A list EOS uses its first element as the n-gram boundary (upstream eos_token_id[0]).
assert.equal(resolveQwen4ExpSourceContracts(text({ eos_token_id: [248046, 248044] })).ngramEmbedding.boundaryTokenId, 248046);

// Conversion: qwen4_exp configs must author every contract exactly; others may not author them.
const authored = JSON.parse(JSON.stringify(contracts));
assert.doesNotThrow(() => assertConvertedQwen4ExpContracts(authored, flashNext, 'flash-next'));
assert.throws(() => assertConvertedQwen4ExpContracts({ ...authored, ngramEmbedding: undefined }, flashNext, 'flash-next'),
  /must author inference.ngramEmbedding/);
const wrongSeed = JSON.parse(JSON.stringify(authored));
wrongSeed.ngramEmbedding.seed = 1235;
assert.throws(() => assertConvertedQwen4ExpContracts(wrongSeed, flashNext, 'flash-next'), /source derives/);
const wrongBoundary = JSON.parse(JSON.stringify(authored));
wrongBoundary.indexedAttention.denseVisibleLimit = 2048;
assert.throws(() => assertConvertedQwen4ExpContracts(wrongBoundary, flashNext, 'flash-next'), /indexedAttention/);
assert.throws(() => assertConvertedQwen4ExpContracts({ hyperConnections: authored.hyperConnections }, { model_type: 'qwen3_5' }, 'other'),
  /not qwen4_exp/);
assert.doesNotThrow(() => assertConvertedQwen4ExpContracts({}, { model_type: 'qwen3_5' }, 'other'));

console.log('qwen4-exp-source.test: ok');
