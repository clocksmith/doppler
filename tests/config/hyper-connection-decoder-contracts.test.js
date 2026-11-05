import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolveHyperConnectionDecoderContracts } from '../../src/config/schema/hyper-connection-decoder.schema.js';
import { deriveNgramLayerTable, deriveNgramMultipliers } from '../../src/config/schema/ngram-embedding.schema.js';
import { resolveQwen4ExpSourceContracts } from '../../src/converter/qwen4-exp-source.js';
import { mergeConfig } from '../../src/config/merge.js';

// Contract: hyper-connection decoder semantics are source-owned. Malformed or
// missing contracts fail before allocation; runtime overlays cannot rewrite
// them; ordinary manifests resolve every contract to null.
const flashNext = JSON.parse(await readFile(
  new URL('../../artifacts/qwen38-flash-next-20261013/source/config.json', import.meta.url), 'utf8'));
const layerTypes = flashNext.text_config.layer_types;
const source = resolveQwen4ExpSourceContracts(flashNext);
const geometry = { hiddenSize: 2560, numLayers: 48, vocabSize: 248320, layerTypes, attentionRotaryDim: 64, modelId: 'fn' };
const session = { rowCache: { maxRows: 0, maxBytes: 0, eviction: 'lru' }, staging: { maxRows: 16 }, reads: { maxConcurrent: 1 } };
const clone = (value) => JSON.parse(JSON.stringify(value));
const base = () => ({
  ...clone(source),
  session: { ngramEmbedding: clone(session) },
  normalization: { postAttentionNorm: false, preFeedforwardNorm: false, postFeedforwardNorm: false },
  layerPattern: { residualBranchScale: 1 },
});
const resolve = (inference, extra = {}) => resolveHyperConnectionDecoderContracts(inference, { ...geometry, ...extra });

const ok = resolve(base());
assert.equal(ok.hyperConnections.streamCount, 4);
assert.equal(ok.indexedAttention.denseVisibleLimit, 2051);
assert.equal(ok.ngramEmbedding.layers[0].decoderLayerIndex, 1);
assert.equal(ok.ngramEmbeddingSession.staging.maxRows, 16);
assert.ok(Object.isFrozen(ok.ngramEmbedding.layers[0]));

// Exact integer derivation: a multiplier off by one in its low bit is rejected.
const lowBit = base();
lowBit.ngramEmbedding.layers[0].multipliers[2] = (BigInt(lowBit.ngramEmbedding.layers[0].multipliers[2]) - 2n).toString();
assert.throws(() => resolve(lowBit), /multipliers\[2\].*derives 8052911324071/);
const asNumber = base();
asNumber.ngramEmbedding.layers[0].multipliers[0] = 23703573157769;
assert.throws(() => resolve(asNumber), /decimal integer string/);
assert.deepEqual(deriveNgramMultipliers(248320, 3, 0, 1234), [23703573157769n, 20109073645365n, 8052911324071n]);
assert.equal(deriveNgramLayerTable({ vocabSizeBase: 20000000, vocabPadMultiple: 128, headCount: 16, pleLayerIndex: 0 }).rows, 320001536);

// One-based source ids translate to zero-based decoder indices.
const offByOne = base();
offByOne.ngramEmbedding.layers[0].decoderLayerIndex = 2;
assert.throws(() => resolve(offByOne), /must equal one-based sourceLayerId 2 minus one/);
const fullLayer = base();
Object.assign(fullLayer.ngramEmbedding.layers[0], { sourceLayerId: 4, decoderLayerIndex: 3 });
assert.throws(() => resolve(fullLayer), /only on linear_attention/);

// Derived QSA geometry and Doppler's declared choices.
for (const [field, value, pattern] of [
  ['denseVisibleLimit', 2048, /tokenBudget \+ compressRatio - 1 = 2051/],
  ['blockTopK', 2048, /tokenBudget \/ compressRatio = 512/],
  ['keyHeads', 2, /exactly one indexer key head/],
  ['forceSelfToken', true, /does not force/],
  ['tieOrder', 'torch_topk', /tieOrder/],
  ['rotaryDim', 128, /attention rotary dimension 64/],
]) {
  const inference = base();
  inference.indexedAttention[field] = value;
  assert.throws(() => resolve(inference), pattern, field);
}
const sparseLayers = base();
sparseLayers.indexedAttention.layers = [3, 7];
assert.throws(() => resolve(sparseLayers), /indexes every full-attention layer/);

// Hyper-connections replace pre-norm residuals and require matching widths.
const width = base();
width.hyperConnections.streamWidth = 2048;
assert.throws(() => resolve(width), /must equal architecture.hiddenSize 2560/);
const sandwich = base();
sandwich.normalization.postAttentionNorm = true;
assert.throws(() => resolve(sandwich), /replace pre-norm residuals/);
const orphan = base();
orphan.hyperConnections = null;
assert.throws(() => resolve(orphan), /without hyperConnections/);

// Unknown and missing fields fail rather than defaulting.
const unknown = base();
unknown.hyperConnections.injectScale = 2;
assert.throws(() => resolve(unknown), /unknown field\(s\) injectScale/);
const missing = base();
delete missing.ngramEmbedding.seed;
assert.throws(() => resolve(missing), /missing required field\(s\) seed/);

// Session policy is required exactly when the table exists.
const noSession = base();
noSession.session = null;
assert.throws(() => resolve(noSession), /session.ngramEmbedding: is required/);
const tinyStaging = base();
tinyStaging.session.ngramEmbedding.staging.maxRows = 15;
assert.throws(() => resolve(tinyStaging), /one token's 16 rows/);
const strayPolicy = { ...base(), ngramEmbedding: null };
assert.throws(() => resolve(strayPolicy), /declares no ngramEmbedding contract/);

// Tensors of an undeclared mechanism fail instead of running as another family.
const none = { normalization: {}, layerPattern: { residualBranchScale: 1 } };
assert.deepEqual(resolve(none, { tensorNames: ['model.language_model.layers.0.mlp.gate.weight'] }),
  { hyperConnections: null, indexedAttention: null, ngramEmbedding: null, ngramEmbeddingSession: null });
assert.throws(() => resolve(none, { tensorNames: ['model.language_model.layers.0.attn_hyper_connection.hc_norm.weight'] }),
  /declares no inference.hyperConnections contract/);
const noPle = { ...base(), ngramEmbedding: null, session: null };
assert.throws(() => resolve(noPle, { tensorNames: ['model.language_model.layers.1.ple.key_proj.weight'] }),
  /declares no inference.ngramEmbedding contract/);

// Merge keeps the contracts manifest-only.
const manifest = { modelId: 'fn', inference: { ...base(), attention: {}, ffn: {}, rope: {}, output: {}, chatTemplate: {} } };
const merged = mergeConfig(manifest, undefined);
assert.equal(merged.inference.hyperConnections.streamCount, 4);
assert.equal(merged._sources.get('inference.ngramEmbedding'), 'manifest');
assert.throws(() => mergeConfig(manifest, { indexedAttention: { ...source.indexedAttention, tokenBudget: 4096 } }),
  /runtime.inference.indexedAttention is source-owned/);
assert.equal(mergeConfig({ modelId: 'plain', inference: { attention: {}, normalization: {}, ffn: {}, rope: {}, output: {} } }).inference.ngramEmbedding, null);

console.log('hyper-connection-decoder-contracts.test: ok');
