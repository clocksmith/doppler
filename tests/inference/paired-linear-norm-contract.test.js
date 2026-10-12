import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/inference/pipelines/text/linear-attention/plan.js', import.meta.url).href;
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === '../../../../memory/buffer-pool.js') {
    return { url: `data:text/javascript,${encodeURIComponent(`
      export const acquireBuffer=(size)=>({size});
      export const uploadData=(buffer,values)=>{buffer.values=values.slice();};
      export const releaseBuffer=()=>{};
      export const readBuffer=()=>{throw Error('unexpected readback');};
    `)}`, shortCircuit: true };
  }
  return next(specifier, context);
} });
try {
  const { createLayerRuntimeState } = await import(target);
  const norm = new Float32Array([0.75, 1.25]);
  const weights = { linearConv1D: new Float32Array(24), linearDtBias: new Float32Array(1),
    linearALog: new Float32Array(1), linearNorm: norm };
  const geometry = { convDim: 6, keyDim: 2, valueDim: 2, numKHeads: 1, numVHeads: 1,
    headKDim: 2, headVDim: 2, qSize: 2, kSize: 2, vSize: 2, qRep: 1 };
  for (const precision of ['f32', 'two-f32']) {
    for (const globalOffset of [false, true]) {
      const state = await createLayerRuntimeState(0, weights, { linearConvKernelDim: 4,
        linearNormMode: 'shared', rmsNormEps: 1e-6, rmsNormWeightOffset: globalOffset,
        sessionSettings: { intermediatePrecision: precision, linearRecurrentStatePrecision: precision } }, 0, geometry);
      assert.deepEqual(state.normWeight, norm, 'linear norm uses exact direct weights');
      assert.deepEqual(state.normWeightGPU.values, norm, 'uploaded direct weights remain unchanged');
      assert.equal(state.normWeightOffset, false, 'surrounding RMSNorm offset must not alter linear output normalization');
    }
  }
} finally { hooks.deregister(); }
console.log('scalar and paired linear normalization preserve direct source weights independently of surrounding norm offset');
