import assert from 'node:assert/strict';
import { assertRecurrentStateLayout, resolveRecurrentStateLayout } from '../../src/inference/pipelines/text/linear-attention/state-layout.js';
import { createLinearAttentionRuntime, cloneLinearAttentionRuntime, restoreLinearAttentionRuntime, resetLinearAttentionRuntime } from '../../src/inference/pipelines/text/linear-attention.js';
import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';

function state(precision) {
  const components = precision === 'f32' ? 1 : 2;
  return { layerIdx: 3, seqLen: 2, numVHeads: 2, headKDim: 2, headVDim: 2,
    recurrentStatePrecision: precision, recurrentStateComponents: components,
    recurrentState: Float32Array.from({ length: 8 * components }, (_, i) => (i + 1) / 32),
    convDim: 1, convKernelSize: 2, convStatePrecision: 'f32', convStateComponents: 1, normWeightOffset: false, convState: new Float32Array([3, 4]), convWeight: new Float32Array([1]), dtBias: new Float32Array([1]),
    aLog: new Float32Array([1]), normWeight: new Float32Array([1]) };
}
try {
  assert.deepEqual(resolveRecurrentStateLayout(), { precision: 'f32', components: 1 });
  for (const precision of ['f32', 'two-f32']) {
    setRuntimeConfig({ inference: { session: { linearRecurrentStatePrecision: precision } } });
    const layer = state(precision);
    assert.equal(assertRecurrentStateLayout(layer), precision === 'f32' ? 1 : 2);
    const runtime = createLinearAttentionRuntime(); runtime.layers.set(3, layer);
    const copy = await cloneLinearAttentionRuntime(runtime);
    assert.equal(copy.schemaVersion, 4);
    assert.deepEqual(copy.layers.get(3).recurrentState, layer.recurrentState);
    assert.equal(copy.layers.get(3).recurrentStatePrecision, precision);
    assert.notEqual(copy.layers.get(3).recurrentState, layer.recurrentState);
    const target = createLinearAttentionRuntime(); restoreLinearAttentionRuntime(target, copy);
    copy.layers.get(3).recurrentState.fill(0);
    assert.deepEqual(target.layers.get(3).recurrentState, layer.recurrentState);
    const before = target.layers;
    assert.throws(() => restoreLinearAttentionRuntime(target, { ...copy, schemaVersion: 2 }), /schema version/);
    assert.equal(target.layers, before, 'invalid restore leaves active ownership unchanged');
    const malformed = state(precision); malformed.recurrentStateComponents = 99;
    assert.throws(() => restoreLinearAttentionRuntime(target, { schemaVersion: 4, layers: new Map([[3, malformed]]) }), /storage/);
    assert.equal(target.layers, before);
    resetLinearAttentionRuntime(target); assert.equal(target.layers.size, 0);
  }
  const paired = state('two-f32');
  paired.convStatePrecision = 'two-f32'; paired.convStateComponents = 2;
  paired.convState = new Float32Array([1, 2 ** -25, 3, -(2 ** -24)]);
  const original = createLinearAttentionRuntime(); original.layers.set(3, paired);
  const cloned = await cloneLinearAttentionRuntime(original);
  assert.deepEqual(cloned.layers.get(3).convState, paired.convState);
  assert.equal(cloned.layers.get(3).convStateComponents, 2);
  const broken = { ...paired, convState: new Float32Array(2) };
  const target = createLinearAttentionRuntime(); const before = target.layers;
  assert.throws(() => restoreLinearAttentionRuntime(target, { schemaVersion: 4, layers: new Map([[3, broken]]) }), /convolution state storage/);
  assert.equal(target.layers, before);
  assert.throws(() => assertRecurrentStateLayout({ ...state('two-f32'), recurrentState: new Float32Array(8) }), /storage/);
  assert.throws(() => assertRecurrentStateLayout({ ...state('f32'), recurrentStatePrecision: undefined }), /precision/);
  setRuntimeConfig({ inference: { session: { linearRecurrentStatePrecision: 'f64' } } });
  assert.throws(resolveRecurrentStateLayout, /precision/);
} finally { resetRuntimeConfig(); }
console.log('linear-recurrent-layout: passed');
