import assert from 'node:assert/strict';
import { compileWeightRepresentationPlan as compile } from '../../src/loader/weight-representation-plan.js';

const defaults = { useFusedQ4K: true, keepF32Weights: false, q4kLayout: 'row', q4kMaterializationMode: 'mixed' };
const manifest = { architecture: { numLayers: 2 } };
const name = (layer, suffix = 'self_attn.q_proj.weight') => `model.layers.${layer}.${suffix}`;
const tensor = offset => ({ shardIndex: 0, offset, size: 144, shape: [1, 256], dtype: 'Q4_K_M', role: 'matmul' });
const step = (op, dtype) => ({ op, kernel: dtype === 'q4k' ? 'fused_matmul_q4.wgsl'
  : dtype === 'f16' ? 'matmul_f16w_f32a_tiled.wgsl' : 'matmul_f32.wgsl', entry: dtype === 'q4k' ? 'main_multicol' : 'main' });
const path = (dtype = 'q4k') => ({ id: dtype, prefill: { steps: [step('q_proj', dtype)] },
  decode: { steps: [step('q_proj', dtype)] }, preLayer: [], postLayer: [] });
const locations = new Map([[name(0), tensor(0)], [name(1), tensor(144)]]);

// An override changes only its actual layer; a role-wide union would add two shadows.
const selected = path();
selected.layerOverrides = [{ layers: [1], prefill: { steps: [step('q_proj', 'f16')] } }];
const frozenInput = JSON.stringify(selected);
const plan = compile(locations, manifest, [selected], defaults);
assert.equal(plan[name(0)].mode, 'packed-only');
assert.equal(plan[name(1)].mode, 'both');
assert.deepEqual(plan[name(1)].dtypes, ['f16', 'q4k']);
assert.equal(JSON.stringify(selected), frozenInput);
assert.throws(() => { plan[name(0)].mode = 'both'; }, TypeError);
assert.throws(() => plan[name(0)].dtypes.push('f32'), TypeError);
assert.throws(() => { plan[name(0)].consumers[0].role = 'other'; }, TypeError);

const fallback = compile(locations, manifest, [selected, path('f32')], defaults);
assert.deepEqual(fallback[name(0)].dtypes, ['f32', 'q4k']);
assert.deepEqual(fallback[name(1)].dtypes, ['f16', 'f32', 'q4k']);
assert.equal(fallback[name(1)].mode, 'both');
const dense = compile(locations, manifest, [path('f16'), path('f32')], defaults);
assert.equal(dense[name(0)].mode, 'dense-only');
assert.deepEqual(dense[name(0)].dtypes, ['f16', 'f32']);

// Equal immutable storage unifies demand even across layer bindings.
const aliased = new Map(locations); aliased.set(name(1), tensor(0));
const aliasPlan = compile(aliased, manifest, [selected], defaults);
assert.equal(aliasPlan[name(0)].mode, 'both');
assert.deepEqual(aliasPlan[name(0)].aliases, [name(0), name(1)]);
aliased.set(name(1), { ...tensor(0), shape: [2, 128] });
assert.equal(compile(aliased, manifest, [selected], defaults)[name(0)].mode, 'packed-only');
aliased.set('unclassified-consumer', tensor(0));
assert.equal(compile(aliased, manifest, [selected], defaults)[name(0)].complete, false);
assert.equal(compile(aliased, manifest, [selected], defaults)[name(0)].mode, 'both');

// The prefill head is used only in prefill; decoding does not use that role.
const head = path(); head.postLayer = [step('lm_head', 'q4k'), step('lm_head_prefill', 'f16')];
const endpoints = new Map([['embed', { ...tensor(0), dtype: 'F16', role: 'embedding' }],
  ['head', { ...tensor(144), role: 'lm_head' }]]);
const tied = { ...manifest, inference: { output: { tieWordEmbeddings: true } } };
const endpointPlan = compile(endpoints, tied, [head], defaults);
assert.deepEqual(endpointPlan.head.dtypes, ['f16', 'q4k']);
assert.equal(endpointPlan.head.consumers.length, 2);
assert.equal(endpointPlan.embed.mode, 'dense-only');
assert.deepEqual(endpointPlan.head.aliases, ['head'], 'A separately quantized head is not a tied dense embedding.');
endpoints.set('head', { ...tensor(144), dtype: 'F16', role: 'lm_head' });
assert.deepEqual(compile(endpoints, tied, [head], defaults).head.aliases, ['embed', 'head']);
assert.throws(() => compile(locations, { architecture: {} }, [path()], defaults), /numLayers/);
console.log('weight-representation-plan.test: ok');

const fused = path();
fused.prefill.steps = [step('ffn_gate', 'f16'), step('ffn_gate_up', 'q4k'), step('linear_out_proj', 'f32')];
fused.decode.steps = [step('ffn_gate', 'f16'), step('linear_out_proj', 'q4k')];
const projections = new Map([[name(0, 'mlp.gate_proj.weight'), tensor(0)],
  [name(0, 'linear_attn.out_proj.weight'), tensor(144)]]);
const projectionPlan = compile(projections, manifest, [fused], defaults);
assert.deepEqual(projectionPlan[name(0, 'mlp.gate_proj.weight')].dtypes, ['f16', 'q4k']);
assert.deepEqual(projectionPlan[name(0, 'linear_attn.out_proj.weight')].dtypes, ['f32', 'q4k']);

// A Q4 path explicitly permits stored F16 weights to reach adaptive F16 kernels.
const floats = new Map([[name(0), { ...tensor(0), dtype: 'F16', size: 512 }]]);
const floatFallback = compile(floats, manifest, [path(), path('f32')], defaults)[name(0)];
assert.deepEqual(floatFallback.dtypes, ['f16', 'f32']);
assert.equal(floatFallback.mode, 'dense-only');
assert.equal(floatFallback.primaryDtype, 'f16');
assert.equal(floatFallback.consumers[0].declaredWeightDtype, 'q4k');
assert.equal(floatFallback.consumers[0].weightDtype, 'f16');
