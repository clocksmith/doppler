import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/inference/pipelines/text/moe/qwen-route-executor.js', import.meta.url).href;
const calls = [], released = [];
globalThis.sharedSlotExecutionTest = {
  fail: null,
  project(options) { calls.push(['router', options.phaseOverride]); return { buffer: { label: 'router' } }; },
  gu() { calls.push(['gu']); if (this.fail === 'gu') throw Error('injected GU'); return { buffer: { label: 'activated' } }; },
  down() { calls.push(['down']); if (this.fail === 'down') throw Error('injected down'); return { buffer: { label: 'output' } }; },
  release(recorder, buffer) { released.push([recorder !== null, buffer.label]); },
};
const modules = {
  '../../../../gpu/kernels/matmul.js': `export const runMatmul=async (_a,_b,_m,_n,_k,o)=>sharedSlotExecutionTest.project(o);
    export const recordMatmul=async (_r,...args)=>runMatmul(...args);`,
  '../../../../gpu/kernels/moe-shared-slot.js': `export const executeSharedSlotGateUp=async()=>sharedSlotExecutionTest.gu();
    export const executeSharedSlotDown=async()=>sharedSlotExecutionTest.down();`,
  '../ffn/qwen-shared-expert.js': 'export const validateQwenSharedExpert=()=>{};',
  '../ops.js': 'export const releaseOrTrack=(r,b)=>sharedSlotExecutionTest.release(r,b);',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { runQwenRouteExperts, recordQwenRouteExperts } = await import(target);
  const weight = { buffer: { __dopplerFakeGPUBuffer: true, size: 4096 }, dtype: 'q4_0', layout: 'row', shape: [4,32,64] };
  const options = { inputTensor: { dtype: 'f32', shape: [1,64], buffer: {} }, indicesBuffer: {}, weightsBuffer: {},
    weights: { gate: weight, up: weight, down: weight, numExperts: 4, expertFormat: 'qwen3_5' },
    expectedExpertFormat: 'qwen3_5', activationDtype: 'f32', hiddenActivation: 'silu',
    numTokens: 1, hiddenSize: 64, intermediateSize: 32, topK: 3, swigluLimit: null, layerIdx: 7,
    sharedSlot: { intermediateSize: 32, phase: 'prefill', kernelPath: {}, weights: {
      sharedExpertGate: weight, sharedExpertUp: weight, sharedExpertDown: weight, sharedExpertRouter: weight,
    } } };
  for (const recorded of [false,true]) for (const failure of [null,'gu','down']) {
    calls.length=0; released.length=0; sharedSlotExecutionTest.fail=failure;
    const promise=recorded?recordQwenRouteExperts({},options):runQwenRouteExperts(options);
    if(failure)await assert.rejects(promise,/injected/);
    else assert.equal((await promise).buffer.label,'output');
    assert.deepEqual(calls, failure==='gu'?[['router','prefill'],['gu']]:[['router','prefill'],['gu'],['down']]);
    assert.deepEqual(released, failure==='gu'?[[recorded,'router']]:[[recorded,'activated'],[recorded,'router']]);
  }
  console.log('shared-slot route adapter ownership and explicit phase checks passed (mocked CPU)');
} finally { hooks.deregister(); delete globalThis.sharedSlotExecutionTest; }
