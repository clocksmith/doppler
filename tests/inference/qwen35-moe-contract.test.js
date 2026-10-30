import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
installNodeFileFetchShim();
const { extractArchitecture } = await import('../../src/converter/artifact-identity.js');
const { resolveManifestMoEConfig } = await import('../../src/converter/manifest-builder.js');
const { resolveMoeExecutionProfile, resolveMoeIntermediateSize } = await import('../../src/inference/pipelines/text/moe-shape-validator.js');
const { validateQwenSharedExpert } = await import('../../src/inference/pipelines/text/ffn/qwen-shared-expert.js');

// HF Qwen3.6-35B-A3B source revision 995ad96eacd98c81ed38be0c5b274b04031597b0.
const source = { model_type: 'qwen3_5_moe', text_config: {
  model_type: 'qwen3_5_moe_text', num_hidden_layers: 40, hidden_size: 2048,
  moe_intermediate_size: 512, shared_expert_intermediate_size: 512,
  num_experts: 256, num_experts_per_tok: 8, num_attention_heads: 16,
  num_key_value_heads: 2, head_dim: 256, vocab_size: 248320,
  max_position_embeddings: 262144,
} };
const architecture = extractArchitecture(source);
assert.equal(architecture.intermediateSize, 512, 'MoE source has no dense intermediate_size');
assert.equal(architecture.numLayers, 40);
assert.equal(architecture.hiddenSize, 2048);
const moe = resolveManifestMoEConfig({ tensors: [] }, { modelId: 'qwen-fixture' }, source, 'transformer');
assert.deepEqual(moe, { numExperts: 256, numExpertsPerToken: 8,
  expertFormat: 'qwen3_5', expertIntermediateSize: 512, sharedExpertIntermediateSize: 512 });
const execution = resolveMoeExecutionProfile({ expertFormat: moe.expertFormat, modelType: 'transformer' });
assert.equal(execution.expertExecutor, 'qwen35-packed');
assert.equal(execution.routerScaleMode, 'none');
assert.equal(execution.topkRouteExecutor, 'qwen35-route', 'Qwen route execution does not require Gemma per-expert scales');
assert.equal(resolveMoeIntermediateSize({ intermediateSize: 1024, expertIntermediateSize: 512 }, execution), 512);
assert.throws(() => resolveManifestMoEConfig({ tensors: [] }, {}, {
  ...source, text_config: { ...source.text_config, shared_expert_intermediate_size: undefined },
}, 'qwen3_5_moe_text'), /sharedExpertIntermediateSize/);
assert.throws(() => resolveManifestMoEConfig({ tensors: [] }, {}, {
  ...source, text_config: { ...source.text_config, num_experts_per_tok: 257 },
}, 'qwen3_5_moe_text'), /cannot exceed/);

const tensor = { dtype: 'f32', shape: [2, 4] };
globalThis.GPUBuffer = class {};
const weights = Object.fromEntries(['sharedExpertGate', 'sharedExpertUp', 'sharedExpertDown', 'sharedExpertRouter'].map(k => [k, new GPUBuffer()]));
const opts = { numTokens: 2, hiddenSize: 4, intermediateSize: 3, kernelPath: null, swigluLimit: null };
assert.doesNotThrow(() => validateQwenSharedExpert(tensor, tensor, weights, opts));
assert.throws(() => validateQwenSharedExpert(tensor, tensor, { ...weights, sharedExpertRouter: null }, opts), /missing sharedExpertRouter/);
assert.throws(() => validateQwenSharedExpert(tensor, { ...tensor, shape: [1, 8] }, weights, opts), /geometry/);
assert.throws(() => validateQwenSharedExpert({ ...tensor, dtype: 'f16' }, tensor, weights, opts), /explicit f32/);
assert.throws(() => validateQwenSharedExpert(tensor, tensor, weights, { ...opts, intermediateSize: 0 }), /positive intermediateSize/);

// All packed expert and shared projections bind to the same declared Q4 kernels
// used by execution, so loading cannot eagerly expand the whole expert bank.
const { compileWeightRepresentationPlan } = await import('../../src/loader/weight-representation-plan.js');
const projection = op => ({ op, kernel: 'fused_matmul_q4.wgsl', entry: 'main_multicol' });
const steps = ['gate_proj', 'up_proj', 'down_proj'].map(projection);
const path = { id: 'qwen-fixture-q4', decode: { steps }, prefill: { steps }, preLayer: [], postLayer: [] };
const names = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj',
  'mlp.shared_expert.gate_proj.weight', 'mlp.shared_expert.up_proj.weight', 'mlp.shared_expert.down_proj.weight'];
const locations = new Map(names.map((suffix, i) => [`model.language_model.layers.0.${suffix}`,
  { shardIndex: 0, offset: i * 144, size: 144, shape: [1, 256], dtype: 'Q4_K_M', role: 'expert' }]));
const planned = compileWeightRepresentationPlan(locations, { architecture: { numLayers: 1 } }, [path],
  { useFusedQ4K: true, keepF32Weights: false, q4kLayout: 'row', q4kMaterializationMode: 'mixed' });
for (const name of locations.keys()) {
  assert.equal(planned[name].mode, 'packed-only', `${name} must not create a dense expert shadow`);
  assert.equal(planned[name].complete, true);
}

// Repacking a 3D expert bank must preserve the independent row-block identity:
// no quantization block may accidentally straddle two experts.
const { quantizeToQ4KMRowWise } = await import('../../src/converter/quantizer.js');
const values = Float32Array.from({length: 2*4*256}, (_,i) => Math.sin(i*0.17)*0.7);
const packedBank = quantizeToQ4KMRowWise(values, [2,4,256]).quantized;
const independentlyPacked = new Uint8Array(packedBank.length);
const expert0 = quantizeToQ4KMRowWise(values.subarray(0,4*256),[4,256]).quantized;
const expert1 = quantizeToQ4KMRowWise(values.subarray(4*256),[4,256]).quantized;
independentlyPacked.set(expert0,0); independentlyPacked.set(expert1,expert0.length);
assert.deepEqual(packedBank,independentlyPacked);
assert.equal(packedBank.length,2*4*144);

const { validateManifest } = await import('../../src/formats/rdrr/validation.js');
const manifestErrors = m => validateManifest({ modelId: 'qwen-fixture', moeConfig: m }).errors.filter(x => x.includes('moeConfig'));
assert.deepEqual(manifestErrors(moe), []);
assert.deepEqual(manifestErrors({ ...moe, sharedExpertIntermediateSize: undefined }), ['Invalid moeConfig.sharedExpertIntermediateSize']);
assert.deepEqual(manifestErrors({ ...moe, expertIntermediateSize: 0 }), ['Invalid moeConfig.expertIntermediateSize']);

// Qwen's source router always normalizes its selected probabilities; the generic
// runtime knob must not silently change that math. Other formats keep their
// existing runtime-controlled normalization contract.
const { assertMoeRoutingContract } = await import('../../src/inference/pipelines/text/moe-shape-validator.js');
assert.doesNotThrow(() => assertMoeRoutingContract(execution, { normalizeWeights: true }));
assert.throws(() => assertMoeRoutingContract(execution, { normalizeWeights: false }), /requires normalizeWeights=true/);
assert.throws(() => assertMoeRoutingContract(execution, {}), /requires normalizeWeights=true/);
const mixtral = resolveMoeExecutionProfile({ expertFormat: 'mixtral', modelType: 'transformer' });
assert.doesNotThrow(() => assertMoeRoutingContract(mixtral, { normalizeWeights: false }));
assert.doesNotThrow(() => assertMoeRoutingContract(mixtral, { normalizeWeights: true }));

console.log('qwen35-moe-contract.test: ok');
