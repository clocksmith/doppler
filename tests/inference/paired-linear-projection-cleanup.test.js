import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { readFileSync } from 'node:fs';
import { createTensor } from '../../src/gpu/tensor.js';
const target = new URL('../../src/inference/pipelines/text/linear-attention.js', import.meta.url).href;
const live = new Set(), held = new Set();
const representation = { precision: 'two-f32', components: 2, elementStrideBytes: 8 };
const allocate = label => { const buffer = { size: 48, gpu: true }; live.add(buffer); return createTensor(buffer, 'f32', [1, 6], label, representation); };
globalThis.linearProjectionFailure = {
  fail: '', release(b) { assert(live.delete(b), 'release each temporary exactly once'); },
  project({ role }) { if (this.fail === role) throw Error('injected projection'); return allocate(role); },
  norm() { if (this.fail === 'norm') throw Error('injected norm'); return allocate('norm'); },
  core() { return allocate('core'); },
  probe(stage) { if (stage === this.fail) throw Error('injected probe'); },
  output() { if (this.fail === 'out') throw Error('injected output'); return allocate('out'); },
};
const plan = `
export const releaseOrTrackBuffer=(r,b)=>r?r.trackTemporaryBuffer(b):linearProjectionFailure.release(b);
export const releaseResolvedWeightBuffer=()=>{};
export const isGpuBuffer=b=>b?.gpu===true;
export const normalizeLinearNormMode=x=>x;
export const inferLinearNormModeFromWeight=()=> 'shared';
export const toPositiveInt=x=>Number.isInteger(x)&&x>0?x:null;
export const ensureLayerRuntimeGpuBuffers=()=>{};
export const createLayerRuntimeState=()=>{throw Error('fixture state required');};
export const applyLinearNormWeightOffset=x=>x;
export const resolveMatmulStepDtype=(_r,_p,_l,_k,d)=>d;
export const projectLinearTensor=args=>linearProjectionFailure.project(args);
export const applyProjectionLoRA=({baseOutput})=>baseOutput;
`;
const modules = {
  './linear-attention/plan.js': plan,
  '../../../gpu/kernels/rmsnorm.js': 'export const runRMSNorm=()=>linearProjectionFailure.norm(); export const recordRMSNorm=runRMSNorm;',
  '../../../gpu/kernels/matmul.js': 'export const runMatmul=()=>linearProjectionFailure.output(); export const recordMatmul=runMatmul;',
  '../../../gpu/kernels/cast.js': `export const castF32ToF16=()=>{throw Error('unexpected cast');};export const castF16ToF32=castF32ToF16;
    export const recordCastF32ToF16=castF32ToF16;export const recordCastF16ToF32=castF32ToF16;`,
  './linear-attention-ab-fusion.js': 'export const resolveLinearAttentionABProjection=()=>null;export const resolveLinearAttentionQKVZProjection=()=>null;',
  '../../../gpu/kernels/linear-attention-core.js': 'export const runLinearAttentionCoreGPU=async()=>linearProjectionFailure.core();',
  './probes.js': 'export const runProbes=async stage=>linearProjectionFailure.probe(stage);',
};
const hooks = registerHooks({ load(url, context, next) {
  if (url === target && process.env.LINEAR_SOURCE_OVERRIDE) return { format: 'module', source: readFileSync(process.env.LINEAR_SOURCE_OVERRIDE, 'utf8'), shortCircuit: true };
  return next(url, context);
}, resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { runLinearAttentionLayer, createLinearAttentionRuntime } = await import(target);
  const input = createTensor({ size: 48 }, 'f32', [1, 6], 'borrowed', representation);
  const weight = { gpu: true }, weights = { qkvProj: weight, oProj: weight, inputNorm: weight, linearInProjZ: weight, linearInProjA: weight, linearInProjB: weight };
  const state = { layerIdx: 0, seqLen: 0, convDim: 6, convKernelSize: 4, keyDim: 2, valueDim: 2, numKHeads: 1, numVHeads: 1, headKDim: 2, headVDim: 2,
    qSize: 2, kSize: 2, vSize: 2, qRep: 1, normMode: 'shared', rmsNormEps: 1e-6,
    convStatePrecision: 'two-f32', convStateComponents: 2, convState: new Float32Array(48), normWeightOffset: false,
    recurrentStatePrecision: 'two-f32', recurrentStateComponents: 2, recurrentState: new Float32Array(8) };
  const runtime = createLinearAttentionRuntime(); runtime.layers.set(0, state);
  const config = { linearNumKeyHeads: 1, linearNumValueHeads: 1, linearKeyHeadDim: 2, linearValueHeadDim: 2, linearNormMode: 'shared',
    rmsNormEps: 1e-6, rmsNormWeightOffset: false, inputDtype: 'f32', outputDtype: 'f32', sessionSettings: { intermediatePrecision: 'two-f32', linearRecurrentStatePrecision: 'two-f32' } };
  for (const recorded of [false, true]) {
    const recorder = recorded ? { trackTemporaryBuffer: b => held.add(b) } : null;
    for (const failure of ['norm', 'linear_qkv_proj', 'linear_z_proj', 'linear_a_proj', 'linear_b_proj', 'linear_core_out', 'out']) {
      state.seqLen = 0; linearProjectionFailure.fail = failure;
      await assert.rejects(runLinearAttentionLayer(input, weights, { layerIdx: 0, numTokens: 1, hiddenSize: 6, config, currentSeqLen: 0,
        activationDtype: 'f32', linearRuntime: runtime, getWeightBuffer: w => w, getNormWeightBuffer: w => w, recorder }), /injected/);
      for (const b of held) linearProjectionFailure.release(b); held.clear();
      assert.equal(live.size, 0, `${failure}: every produced projection/core/norm temporary settled`);
    }
  }
} finally { hooks.deregister(); delete globalThis.linearProjectionFailure; }
console.log('paired linear layer: normalization, each projection, core observation and output failure cleanup passed');
