import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createKernelRegistry } from '../../src/config/kernel-registry-contract.js';

const baseline = createKernelRegistry();
const variant = baseline.getKernelConfig('scale', 'default');
const extension = mode => ({ extensions: { scale: { variants: { completion_control: {
  wgsl: variant.shaderFile, entryPoint: variant.entryPoint, workgroup: [64, 1, 1],
  immediateCompletion: mode,
} } } } });
for (const invalid of [null, '', 'wait', false, 1]) {
  assert.throws(() => createKernelRegistry(extension(invalid)), /invalid immediateCompletion/);
}
const submitted = createKernelRegistry(extension('submitted'));
const settled = createKernelRegistry(extension('settled'));
assert.notEqual(submitted.identity, settled.identity);
assert.equal(variant.immediateCompletion, 'submitted');
assert.equal(baseline.getKernelConfig('matmul', 'paired_q4k').immediateCompletion, 'settled');
assert(Object.isFrozen(settled.getKernelConfig('scale', 'completion_control')));

const target = new URL('../../src/gpu/kernels/kernel-execution.js', import.meta.url).href;
const context = globalThis.completionControl = {};
const mocks = {
  './kernel-configs.js': 'export const getKernelRegistry=()=>({getKernelConfig:()=>completionControl.config,getKernelValidator:()=>null});',
  './pipeline-cache.js': 'export const getCachedPipeline=()=>({}); export const getPipelineFast=async()=>({}); export const getPipelineBindGroupLayout=()=>({});',
  '../device.js': 'export const getDevice=()=>completionControl.device;',
  './dispatch.js': 'const submit=()=>{completionControl.dispatches++;if(completionControl.submitError)throw completionControl.submitError;};export const dispatchKernel=submit,dispatchIndirect=submit,recordDispatchIndirect=submit;',
  './uniform-utils.js': 'export const createKernelUniformBuffer=(_a,_b,_c,recorder)=>{const b={destroy:()=>completionControl.destroyed++};if(recorder)completionControl.recorderOwned.push(b);return b;};',
  '../../memory/buffer-pool.js': 'export const acquireBuffer=()=>{throw Error("unexpected allocation")};export const releaseBuffer=()=>{};',
};
const hooks = registerHooks({ resolve(specifier, parent, next) {
  if (parent.parentURL === target && mocks[specifier]) {
    return { url: `data:text/javascript,${encodeURIComponent(mocks[specifier])}`, shortCircuit: true };
  }
  return next(specifier, parent);
} });
function reset(mode, uniforms = true) {
  let resolve, reject;
  const completion = new Promise((yes, no) => { resolve = yes; reject = no; });
  Object.assign(context, { config: { immediateCompletion: mode, uniforms: uniforms ? {} : null,
    bindings: uniforms ? [{index:0,type:'uniform'}] : [] },
    dispatches:0, waits:0, destroyed:0, recorderOwned:[], submitError:null,
    device: { createBindGroup:()=>({}), queue: { onSubmittedWorkDone() { context.waits++; return completion; } } } });
  return { resolve, reject };
}
try {
  const { unifiedKernelWrapper } = await import(target);
  const run = (target = null, groups = [1,1,1], signal = null) => unifiedKernelWrapper(
    'control', target, 'default', [], {}, groups, null, null, null, signal);
  for (const uniforms of [false,true]) for (const indirect of [false,true]) {
    const gate=reset('settled',uniforms);let returned=false;
    const pending=run(null,indirect ? {indirectBuffer:{}} : [1,1,1]).then(v=>{returned=true;return v;});
    await Promise.resolve();assert.equal(context.dispatches,1);assert.equal(context.waits,1);
    assert.equal(returned,false);assert.equal(context.destroyed,0);
    gate.resolve();assert.equal(await pending,true);assert.equal(context.destroyed,Number(uniforms));
  }
  {
    const gate=reset('settled');const failure=Error('device completion rejected');
    const pending=run();gate.reject(failure);await assert.rejects(pending,error=>error===failure);
    assert.equal(context.destroyed,1);
  }
  {
    const gate=reset('settled');const abort=new AbortController();let returned=false;
    const pending=run(null,[1,1,1],abort.signal).finally(()=>{returned=true;});
    abort.abort(Error('cancel after dispatch'));await Promise.resolve();assert.equal(returned,false);
    assert.equal(context.destroyed,0);gate.resolve();await assert.rejects(pending,/cancel after dispatch/);
    assert.equal(context.destroyed,1);
  }
  {
    reset('settled');context.submitError=Error('submit rejected');
    await assert.rejects(run(),/submit rejected/);assert.equal(context.waits,0);assert.equal(context.destroyed,1);
  }
  {
    const gate=reset('submitted');assert.equal(await run(),true);assert.equal(context.destroyed,0);
    gate.resolve();await Promise.resolve();assert.equal(context.destroyed,1);
  }
  {
    reset('settled');const recorder={device:context.device,beginComputePass(){}};
    assert.equal(await run(recorder),true);assert.equal(context.waits,0);assert.equal(context.destroyed,0);
    assert.equal(context.recorderOwned.length,1);context.recorderOwned[0].destroy();
  }
} finally { hooks.deregister();delete globalThis.completionControl; }
console.log('kernel completion policy: identity, validation, immediate/recorded ownership, cancellation and rejection passed');
