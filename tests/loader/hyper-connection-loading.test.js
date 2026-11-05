import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { loadHyperConnectionLayerWeights, loadHyperConnectionMixerWeights } from '../../src/loader/hyper-connection-loader.js';
import { createWeightBuffer, tagBufferDtype } from '../../src/gpu/weight-buffer.js';
import { compileWeightRepresentationPlan } from '../../src/loader/weight-representation-plan.js';
import { loadFinalWeights } from '../../src/loader/final-weights-loader.js';

const hyper = { semantics: 'qwen4_exp_gated_residual', streamCount: 2, streamWidth: 4, projectionRank: 3, normEps: 1e-6 };
const indexed = { layers: [1], queryHeads: 2, keyHeads: 1, headDim: 2 };
const ngram = { layers: [{ decoderLayerIndex: 1 }], embedDim: 4, convKernelSize: 4 };
const root = 'model.language_model';

function fixture() {
  const tensorLocations = new Map();
  const loaded = new Map();
  const calls = [];
  function add(name, shape, role = 'matmul') {
    const count = shape.reduce((a, b) => a * b, 1);
    const dtype = role === 'matmul' ? 'f16' : 'f32';
    const buffer = { __dopplerFakeGPUBuffer: true, size: count * (dtype === 'f16' ? 2 : 4), destroy() {} };
    tagBufferDtype(buffer, dtype);
    tensorLocations.set(name, { size: buffer.size, shape, dtype: dtype.toUpperCase(), role, layout: 'row' });
    loaded.set(name, role === 'matmul' ? createWeightBuffer(buffer, dtype, 'row', shape, name) : buffer);
  }
  for (const branch of ['attn_hyper_connection', 'mlp_hyper_connection', 'hyper_connection_mixer']) {
    const prefix = branch === 'hyper_connection_mixer' ? `${root}.${branch}` : `${root}.layers.1.${branch}`;
    add(`${prefix}.hc_norm.weight`, [8], 'norm');
    add(`${prefix}.input_mix_weight_down.weight`, [3, 8]);
    add(`${prefix}.input_mix_weight_up.weight`, [8, 3]);
    if (branch !== 'hyper_connection_mixer') add(`${prefix}.block_inject_weight.weight`, [2, 8]);
  }
  add(`${root}.layers.1.self_attn.indexer.index_qk_proj.weight`, [6, 4]);
  add(`${root}.layers.1.self_attn.indexer.q_layernorm.weight`, [2], 'norm');
  add(`${root}.layers.1.self_attn.indexer.k_layernorm.weight`, [2], 'norm');
  add(`${root}.layers.1.ple.key_proj.weight`, [8, 4]);
  add(`${root}.layers.1.ple.value_proj.weight`, [4, 4]);
  for (const name of ['norm_key', 'norm_query', 'norm_conv']) add(`${root}.layers.1.ple.${name}.weight`, [8], 'norm');
  add(`${root}.layers.1.ple.conv1d.weight`, [8, 1, 4], 'conv');
  // The host table is metadata only. Loading small decoder weights must never materialize it.
  tensorLocations.set(`${root}.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight`, {
    shape: [10000000, 2], size: 40000000, dtype: 'BF16', role: 'embedding',
  });
  const ctx = { tensorLocations, hyperConnections: hyper, indexedAttention: indexed, ngramEmbedding: ngram,
    loadTensor: async (name, toGPU, silent, role) => { calls.push({ name, toGPU, silent, role }); return loaded.get(name); } };
  return { ctx, loaded, calls };
}

{
  const { ctx, calls, loaded } = fixture();
  const weights = await loadHyperConnectionLayerWeights(ctx, 1);
  assert.equal(weights.hyperConnections.attention.down, loaded.get(`${root}.layers.1.attn_hyper_connection.input_mix_weight_down.weight`));
  assert.deepEqual(weights.hyperConnections.attention.norm.shape, [8]);
  assert.deepEqual(weights.ple.conv.shape, [32]);
  assert.deepEqual(weights.qsaIndexer.qNorm.shape, [2]);
  assert.equal(calls.length, 17);
  assert.ok(calls.every(call => call.toGPU === true && call.silent === false && !call.name.includes('ngram_embedding')));
  assert.equal(calls.find(call => call.name.endsWith('ple.key_proj.weight')).role, 'ple_key_proj');
  const mixer = await loadHyperConnectionMixerWeights(ctx);
  assert.equal(mixer.inject, null);
  assert.equal(mixer.up, loaded.get(`${root}.hyper_connection_mixer.input_mix_weight_up.weight`));
}
{
  const { ctx, loaded, calls } = fixture();
  const name = `${root}.lm_head.weight`;
  const buffer = { __dopplerFakeGPUBuffer: true, size: 32, destroy() {} };
  ctx.tensorLocations.set(name, { shape: [4, 4], size: 32, dtype: 'F16', role: 'lm_head', group: 'head', layout: 'row' });
  loaded.set(name, createWeightBuffer(buffer, 'f16', 'row', [4, 4], name));
  const result = await loadFinalWeights({ ...ctx, embeddings: null, tieWordEmbeddings: false,
    keepF32Weights: true, gpuBuffers: new Set(), normOffsetDebugLogged: false,
    needsNormWeightOffset: () => true, shouldStreamLargeWeight: () => false, resolveWeightLayout: loc => loc.layout });
  assert.equal(result.finalNorm, null, 'the root mixer replaces the final norm');
  assert.equal(result.hyperConnectionMixer.inject, null);
  assert.equal(result.lmHead, loaded.get(name));
  assert.equal(calls.filter(call => call.name.includes('hyper_connection_mixer')).length, 3);
}
for (const corrupt of ['missing', 'shape', 'duplicate']) {
  const { ctx, calls } = fixture();
  const name = `${root}.layers.1.mlp_hyper_connection.block_inject_weight.weight`;
  if (corrupt === 'missing') ctx.tensorLocations.delete(name);
  if (corrupt === 'shape') ctx.tensorLocations.set(name, { ...ctx.tensorLocations.get(name), shape: [8, 2] });
  if (corrupt === 'duplicate') ctx.tensorLocations.set(name.replace('model.language_model', 'model'), ctx.tensorLocations.get(name));
  await assert.rejects(loadHyperConnectionLayerWeights(ctx, 1), /Expected one|requires shape/);
  assert.equal(calls.length, 0, `${corrupt} rejected before materialization`);
}
{
  const { ctx, calls, loaded } = fixture();
  const name = `${root}.layers.1.attn_hyper_connection.hc_norm.weight`;
  tagBufferDtype(loaded.get(name), 'f16');
  await assert.rejects(loadHyperConnectionLayerWeights(ctx, 1), /complete f32 vector/);
  assert.equal(calls.length, 1);
}
for (const corrupt of ['capacity', 'shape']) {
  const { ctx, loaded } = fixture();
  const name = `${root}.layers.1.attn_hyper_connection.input_mix_weight_down.weight`;
  const weight = loaded.get(name);
  if (corrupt === 'capacity') weight.buffer.size = 2;
  else loaded.set(name, createWeightBuffer(weight.buffer, 'f16', 'row', [8, 3], name));
  await assert.rejects(loadHyperConnectionLayerWeights(ctx, 1), /projection representation/);
}
{
  const { ctx, calls } = fixture();
  ctx.hyperConnections = null;
  assert.deepEqual(await loadHyperConnectionLayerWeights(ctx, 1), {});
  assert.equal(await loadHyperConnectionMixerWeights(ctx), null);
  assert.equal(calls.length, 0);
}
{
  const { ctx } = fixture();
  const roles = ['attn_hc_down', 'attn_hc_up', 'attn_hc_inject', 'mlp_hc_down', 'mlp_hc_up', 'mlp_hc_inject',
    'qsa_index_qk', 'ple_key_proj', 'ple_value_proj', 'hc_mixer_down', 'hc_mixer_up'];
  const steps = roles.map(op => ({ op, kernel: 'matmul_f16w_f32a_tiled.wgsl', entry: 'main' }));
  const layerSteps = steps.filter(step => !step.op.startsWith('hc_mixer_'));
  const postLayer = steps.filter(step => step.op.startsWith('hc_mixer_'));
  const plan = compileWeightRepresentationPlan(ctx.tensorLocations, { architecture: { numLayers: 2 } },
    [{ id: 'hc-test', prefill: { steps: layerSteps }, decode: { steps: layerSteps }, preLayer: [], postLayer }],
    { useFusedQ4K: false, keepF32Weights: false, q4kLayout: 'row', q4kMaterializationMode: 'dense' });
  for (const [name, item] of Object.entries(plan)) {
    if (ctx.tensorLocations.get(name).role !== 'matmul') continue;
    assert.deepEqual(item.dtypes, ['f16'], name);
    assert.equal(item.consumers.length, 2, name);
    assert.equal(item.complete, true, name);
  }
}
{
  // Check the aliases and geometry against the independently recorded real checkpoint headers.
  // Handles are contract doubles: neither source bytes nor a physical GPU are used.
  const receipt = JSON.parse(readFileSync(new URL('../../artifacts/qwen38-flash-next-20261013/checkpoint-headers.json', import.meta.url)));
  const config = JSON.parse(readFileSync(new URL('../../artifacts/qwen38-flash-next-20261013/source/config.json', import.meta.url))).text_config;
  const tensorLocations = new Map(Object.values(receipt.files).flatMap(file => Object.entries(file.tensors)));
  const calls = [];
  const ctx = { tensorLocations,
    hyperConnections: { streamCount: config.hc_count, streamWidth: config.hidden_size, projectionRank: config.hc_lowrank },
    indexedAttention: { layers: config.layer_types.flatMap((kind, i) => kind === 'full_attention' ? [i] : []),
      queryHeads: config.indexer_n_heads, keyHeads: config.indexer_kv_heads, headDim: config.indexer_head_dim },
    ngramEmbedding: { layers: config.ple_layer_ids.map(id => ({ decoderLayerIndex: id - 1 })),
      embedDim: config.ple_embed_dim, convKernelSize: config.ple_conv_kernel_size },
    loadTensor: async name => {
      assert.ok(!name.includes('ngram_embedding'), 'host table cannot be materialized as a decoder weight');
      calls.push(name);
      const { shape } = tensorLocations.get(name);
      const projection = shape.length === 2;
      const dtype = projection ? 'f16' : 'f32';
      const buffer = { __dopplerFakeGPUBuffer: true, size: shape.reduce((a, b) => a * b, 1) * (projection ? 2 : 4), destroy() {} };
      tagBufferDtype(buffer, dtype);
      return projection ? createWeightBuffer(buffer, dtype, 'row', shape, name) : buffer;
    } };
  for (const layer of [0, 1, 3]) {
    const weights = await loadHyperConnectionLayerWeights(ctx, layer);
    assert.equal(Boolean(weights.ple), layer === 1);
    assert.equal(Boolean(weights.qsaIndexer), layer === 3);
  }
  assert.ok(await loadHyperConnectionMixerWeights(ctx));
  assert.equal(calls.length, 36);
}
console.log('hyper-connection-loading.test: ok (contract doubles; no physical GPU)');
