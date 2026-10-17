import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createTensor } from '../../src/gpu/tensor.js';

const target = new URL('../../src/inference/pipelines/text/linear-attention.js', import.meta.url).href;
const released = new Set();
let creations = 0;
let failUpload = false;
const makeBuffer = () => ({ gpu: true });
const geometry = { convDim: 6, keyDim: 2, valueDim: 2, numKHeads: 1, numVHeads: 1,
  headKDim: 2, headVDim: 2, qSize: 2, kSize: 2, vSize: 2, qRep: 1 };
globalThis.linearPreparedTest = {
  release(buffer) { assert(!released.has(buffer), 'buffer released once'); released.add(buffer); },
  upload(buffer, data) { if (failUpload) throw Error('upload failed'); buffer.data = data.slice(); },
  create(layerIdx, weights, config, seqLen) {
    creations++;
    const components = config.sessionSettings.intermediatePrecision === 'two-f32' ? 2 : 1;
    const state = { ...geometry, layerIdx, seqLen, warnedSeqMismatch: false, convKernelSize: 4,
      normMode: 'shared', rmsNormEps: config.rmsNormEps, normWeightOffset: false,
      convStatePrecision: config.sessionSettings.intermediatePrecision, convStateComponents: components,
      recurrentStatePrecision: config.sessionSettings.linearRecurrentStatePrecision,
      recurrentStateComponents: components, convState: new Float32Array(24 * components),
      recurrentState: new Float32Array(4 * components), convWeight: new Float32Array([1]),
      dtBias: new Float32Array([2]), aLog: new Float32Array([3]), normWeight: new Float32Array([4]) };
    for (const field of ['convWeight', 'dtBias', 'aLog', 'normWeight', 'convState', 'recurrentState']) {
      state[`${field}GPU`] = makeBuffer();
    }
    return state;
  },
};
const plan = `
export const isGpuBuffer=b=>b?.gpu===true;
export const toPositiveInt=v=>Number.isInteger(v)&&v>0?v:null;
export const normalizeLinearNormMode=v=>v??null;
export const createLayerRuntimeState=(...args)=>linearPreparedTest.create(...args);
export const ensureLayerRuntimeGpuBuffers=()=>{};
export const releaseOrTrackBuffer=()=>{};
export const releaseResolvedWeightBuffer=()=>{};
export const inferLinearNormModeFromWeight=()=>null;
export const resolveMatmulStepDtype=(_r,_p,_l,_k,d)=>d;
export const applyLinearNormWeightOffset=v=>v;
export const applyProjectionLoRA=()=>{};
export const projectLinearTensor=()=>{throw Error('stop after preparation');};
`;
const modules = {
  './linear-attention/plan.js': plan,
  '../../../memory/buffer-pool.js': `
export const releaseBuffer=b=>linearPreparedTest.release(b);
export const uploadData=(b,d)=>linearPreparedTest.upload(b,d);
export const readBuffer=()=>{throw Error('unexpected readback');};`,
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) {
    return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  }
  return next(specifier, context);
} });
try {
  const { createLinearAttentionRuntime, resetLinearAttentionSequence, resetLinearAttentionRuntime,
    runLinearAttentionLayer } = await import(target);
  const weight = makeBuffer();
  const weights = { qkvProj: weight, oProj: weight, linearInProjZ: weight, linearInProjA: weight,
    linearInProjB: weight, linearConv1D: weight, linearDtBias: weight, linearALog: weight, linearNorm: weight };
  const config = { linearNumKeyHeads: 1, linearNumValueHeads: 1, linearKeyHeadDim: 2,
    linearValueHeadDim: 2, linearConvKernelDim: 4, linearNormMode: 'shared', rmsNormEps: 1e-6,
    rmsNormWeightOffset: false, sessionSettings: { intermediatePrecision: 'f32', linearRecurrentStatePrecision: 'f32' } };
  const runtime = createLinearAttentionRuntime();
  const attempt = async () => {
    const paired = config.sessionSettings.intermediatePrecision === 'two-f32';
    const input = createTensor({ size: paired ? 48 : 24 }, 'f32', [1, 6], 'input',
      paired ? { precision: 'two-f32', components: 2, elementStrideBytes: 8 } : undefined);
    await assert.rejects(runLinearAttentionLayer(input, weights, { layerIdx: 0, numTokens: 1,
      hiddenSize: 6, currentSeqLen: 0, config, linearRuntime: runtime, activationDtype: 'f32',
      getWeightBuffer: w => w, getNormWeightBuffer: w => w }), /stop after preparation/);
  };
  await attempt();
  const first = runtime.layers.get(0);
  first.seqLen = 10; first.warnedSeqMismatch = true;
  first.convState.fill(7); first.recurrentState.fill(9);
  const parameters = first.convWeightGPU;
  resetLinearAttentionSequence(runtime);
  assert.equal(runtime.layers.get(0), first);
  assert.equal(first.seqLen, 0);
  assert.equal(first.warnedSeqMismatch, false);
  for (const field of ['convState', 'recurrentState']) {
    assert(first[field].every(v => v === 0));
    assert(first[`${field}GPU`].data.every(v => v === 0));
  }
  assert.equal(first.convWeightGPU, parameters);
  assert.deepEqual(first.normWeight, new Float32Array([4]));
  await attempt();
  assert.equal(creations, 1, 'fresh request reuses prepared parameters');
  assert.equal(released.size, 0);

  weights.linearNorm = makeBuffer();
  resetLinearAttentionSequence(runtime);
  await attempt();
  assert.equal(creations, 2, 'replaced parameter is materialized again');
  assert.equal(released.size, 6, 'old prepared and dynamic buffers released');
  config.sessionSettings = { intermediatePrecision: 'two-f32', linearRecurrentStatePrecision: 'two-f32' };
  resetLinearAttentionSequence(runtime);
  await attempt();
  assert.equal(creations, 3, 'fresh request with new precision rebuilds state');
  assert.equal(runtime.layers.get(0).convStateComponents, 2);
  config.rmsNormEps = 1e-5;
  resetLinearAttentionSequence(runtime);
  await attempt();
  assert.equal(creations, 4, 'changed preparation settings rebuild state');
  failUpload = true;
  assert.throws(() => resetLinearAttentionSequence(runtime), /upload failed/);
  assert.equal(runtime.layers.size, 0, 'failed reset cannot leave stale conversation state reusable');
  assert.equal(released.size, 24);
  failUpload = false;
  await attempt();
  resetLinearAttentionRuntime(runtime);
  assert.equal(runtime.layers.size, 0, 'full reset still releases prepared capacity');
  assert.equal(released.size, 30);
} finally {
  hooks.deregister();
  delete globalThis.linearPreparedTest;
}
console.log('linear prepared sequence: reuse, isolation, source/precision invalidation, failed reset and full release passed');
