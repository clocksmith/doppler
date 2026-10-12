import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createTensor } from '../../src/gpu/tensor.js';
import { resolveLayerPipeline } from '../../src/inference/pipelines/text/layer-plan.js';

const target = new URL('../../src/inference/pipelines/text/layer-plan-gpu.js', import.meta.url).href;
const live = new Set(), pending = new Set();
const pair = { precision: 'two-f32', components: 2, elementStrideBytes: 8 };
globalThis.pairedPlanControl = {
  fail: null,
  weight() { const buffer = { size: 16 }; live.add(buffer); return buffer; },
  release(buffer) { assert(live.delete(buffer), 'each owned buffer released once'); },
  retain(recorder, buffer) { if (recorder) pending.add(buffer); else this.release(buffer); },
  norm(input, weight) {
    assert.equal(input.precision, 'two-f32'); assert.equal(weight.dtype, 'f32');
    if (this.fail === 'dispatch') throw Error('injected norm dispatch failure');
    const buffer = { size: 32 }; live.add(buffer);
    return createTensor(buffer, 'f32', [1, 4], 'result', this.fail === 'metadata' ? undefined : pair);
  },
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL !== target) return next(specifier, context);
  let source;
  if (specifier === '../../../gpu/device.js') source = 'export const getDevice=()=>({});';
  if (specifier === '../../../memory/buffer-pool.js') source = 'export const releaseBuffer=b=>pairedPlanControl.release(b);';
  if (specifier === './weights.js') source = 'export const getNormWeightBuffer=()=>pairedPlanControl.weight(); export const getWeightBuffer=x=>x;';
  if (specifier === './probes.js') source = 'export const runProbes=async(stage,buffer,options)=>{if(options.precision!=="two-f32")throw Error("lost probe metadata")};';
  if (specifier === './ops.js') source = `
    export const doRMSNorm=(...args)=>pairedPlanControl.norm(...args);
    export const releaseOrTrack=(r,b)=>pairedPlanControl.retain(r,b);
    export const doAttention=()=>{}, doLayerNorm=()=>{}, doResidualAdd=()=>{}, doConv=()=>{}, doCast=()=>{};`;
  return source ? { url: 'data:text/javascript,' + encodeURIComponent(source), shortCircuit: true } : next(specifier, context);
} });
try {
  const { processLayerPlanGPU } = await import(target);
  const input = { size: 32 };
  const step = { op: 'rmsnorm', src: 'state', dst: 'state', weight: 'input' };
  for (const recorded of [false, true]) {
    const recorder = recorded ? { device: {}, trackTemporaryBuffer: b => pending.add(b) } : null;
    const context = { config: { hiddenSize: 4, rmsNormEps: 1e-6 }, weightConfig: { rmsNormWeightOffset: false },
      activationDtype: 'f32', activationPrecision: pair.precision, activationComponents: pair.components,
      activationElementStrideBytes: pair.elementStrideBytes, finitenessGuardEnabled: false,
      pipelinePlan: resolveLayerPipeline({ steps: [step] }, null, 1), recorder };
    for (const failure of ['metadata', 'dispatch']) {
      pairedPlanControl.fail = failure;
      await assert.rejects(processLayerPlanGPU(0, input, 1, true, 4, context, { inputNorm: new Float32Array(4) }, {}),
        failure === 'metadata' ? /discarded paired/ : /injected/);
      for (const buffer of pending) pairedPlanControl.release(buffer);
      pending.clear(); assert.equal(live.size, 0);
    }
    pairedPlanControl.fail = null;
    const invalid = { ...context, pipelinePlan: resolveLayerPipeline({ steps: [{ ...step, inputDtype: 'f16' }] }, null, 1) };
    await assert.rejects(processLayerPlanGPU(0, input, 1, true, 4, invalid, { inputNorm: new Float32Array(4) }, {}), /dtype mismatch/);
    assert.equal(live.size, 0, 'input contract failure occurs before uploading norm weights');
  }
} finally { hooks.deregister(); delete globalThis.pairedPlanControl; }
console.log('paired layer plan rejects lost metadata and settles immediate/recorded failures');
