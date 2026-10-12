import assert from 'node:assert/strict';
import { assertRecurrentStateLayout, resolveRecurrentStateLayout } from '../../src/inference/pipelines/text/linear-attention/state-layout.js';
import { createLinearAttentionRuntime, cloneLinearAttentionRuntime, restoreLinearAttentionRuntime, resetLinearAttentionRuntime } from '../../src/inference/pipelines/text/linear-attention.js';
import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';

function state(precision) {
  const components = precision === 'f32' ? 1 : 2;
  return { layerIdx: 3, seqLen: 2, numVHeads: 2, headKDim: 2, headVDim: 2,
    recurrentStatePrecision: precision, recurrentStateComponents: components,
    recurrentState: Float32Array.from({ length: 8 * components }, (_, i) => (i + 1) / 32),
    convState: new Float32Array([3, 4]), convWeight: new Float32Array([1]), dtBias: new Float32Array([1]),
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
    assert.equal(copy.schemaVersion, 3);
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
    assert.throws(() => restoreLinearAttentionRuntime(target, { schemaVersion: 3, layers: new Map([[3, malformed]]) }), /storage/);
    assert.equal(target.layers, before);
    resetLinearAttentionRuntime(target); assert.equal(target.layers.size, 0);
  }
  assert.throws(() => assertRecurrentStateLayout({ ...state('two-f32'), recurrentState: new Float32Array(8) }), /storage/);
  assert.throws(() => assertRecurrentStateLayout({ ...state('f32'), recurrentStatePrecision: undefined }), /precision/);
  setRuntimeConfig({ inference: { session: { linearRecurrentStatePrecision: 'f64' } } });
  assert.throws(resolveRecurrentStateLayout, /precision/);
} finally { resetRuntimeConfig(); }
console.log('linear-recurrent-layout: passed');
