import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';

const target = new URL('../../src/inference/pipelines/text/hyper-connection/decoder-layer.js', import.meta.url).href;
const state = { events: [], outputs: new Map(), failInjection: false };
const tensor = (label, buffer = { size: 256, label }) => ({ dtype: 'f32', shape: [1, 16], buffer });
globalThis.hcMixBranchWiring = { state, tensor };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier.endsWith('/hyper-connection-up.js')) {
    const source = `const {state:s,tensor:t}=globalThis.hcMixBranchWiring;
      export const recordHyperConnectionUpMixBranch=async(_r,_x,_w,_n,o)=>{s.events.push('up-fused'); return t('up-branch',o.outputBuffer);};
      export const runHyperConnectionUpMixBranch=(x,w,n,o)=>recordHyperConnectionUpMixBranch(null,x,w,n,o);`;
    return { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true };
  }
  if (context.parentURL !== target || !specifier.endsWith('/hyper-connection.js')) return next(specifier, context);
  const source = `const {state:s,tensor:t}=globalThis.hcMixBranchWiring;
    const norm=async()=>t('norm');
    const activate=async(_r,x,o)=>{s.events.push(o.mode); if(s.failInjection && o.mode==='inject_gate')throw Error('injection failed'); return t(o.mode);};
    const branch=async(_r,_x,_n,o)=>{s.events.push('separate');return t('branch',o.outputBuffer);};
    const mix=async(_r,_x,_n,o)=>{s.events.push('fused');return t('branch',o.outputBuffer);};
    export const recordHyperConnectionNorm=norm, runHyperConnectionNorm=norm;
    export const recordHyperConnectionActivation=activate, runHyperConnectionActivation=(x,o)=>activate(null,x,o);
    export const recordHyperConnectionBranch=branch, runHyperConnectionBranch=(x,n,o)=>branch(null,x,n,o);
    export const recordHyperConnectionMixBranch=mix, runHyperConnectionMixBranch=(x,n,o)=>mix(null,x,n,o);
    export const recordHyperConnectionInject=async()=>t('inject'), runHyperConnectionInject=recordHyperConnectionInject;
    export const recordHyperConnectionReplicate=async()=>t('replicate'), runHyperConnectionReplicate=recordHyperConnectionReplicate;`;
  return { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true };
} });
try {
  const { runGatedResidual, runRootMixer } = await import(target);
  const recorder = { beginComputePass() {}, trackTemporaryBuffer(buffer) { state.events.push(`release:${buffer.label}`); },
    decodeWorkspace: { getBuffer(slot, bytes) {
      if (!state.outputs.has(slot)) state.outputs.set(slot, { label: slot, size: bytes });
      return state.outputs.get(slot);
    } } };
  const input = { streams: tensor('streams'), weights: { norm: tensor('norm-weight'), down: {}, up: {}, inject: {} },
    hyper: { streamCount: 4, streamWidth: 4, projectionRank: 2, normEps: 1e-6, normWeightOffset: false },
    numTokens: 1, layerIdx: 3, role: 'attn_hc', project: async role => { state.events.push(`project:${role}`); return tensor(role); } };
  const original = await runGatedResidual(recorder, input);
  assert.ok(state.events.includes('mix_sigmoid') && state.events.includes('separate'));
  assert.equal(state.outputs.size, 3, 'prepared decode owns three intermediate slots independently of mix fusion');
  assert.ok(original.branchInput);
  state.events.length = 0;
  const first = await runGatedResidual(recorder, { ...input, mixBranch: true });
  assert.ok(state.events.includes('fused'));
  assert.ok(!state.events.includes('mix_sigmoid') && !state.events.includes('separate'));
  const second = await runGatedResidual(recorder, { ...input, mixBranch: true });
  assert.equal(first.branchInput.buffer, second.branchInput.buffer, 'serial token calls reuse the same branch slot');
  const mlp = await runGatedResidual(recorder, { ...input, mixBranch: true, role: 'mlp_hc' });
  assert.notEqual(mlp.branchInput.buffer, first.branchInput.buffer, 'different live branch sites never alias');
  state.failInjection = true;
  await assert.rejects(runGatedResidual(recorder, { ...input, mixBranch: true }), /injection failed/);
  assert.ok(state.events.includes('release:hc:3:attn_hc:branch'), 'failure returns the borrowed output to recorder lifetime handling');
  assert.ok(state.events.includes('release:attn_hc_up'), 'up projection remains retained after a later failure');
  state.failInjection = false;
  state.events.length = 0;
  const up = await runGatedResidual(recorder, { ...input, mixBranch: true, upMixBranch: true });
  assert.ok(state.events.includes('up-fused'));
  assert.ok(!state.events.includes('project:attn_hc_up') && !state.events.includes('fused') && !state.events.includes('mix_sigmoid'));
  assert.equal(up.branchInput.buffer, first.branchInput.buffer, 'new producer borrows the same site-scoped branch slot');
  state.failInjection = true;
  await assert.rejects(runGatedResidual(recorder, { ...input, upMixBranch: true }), /injection failed/);
  assert.ok(state.events.includes('release:hc:3:attn_hc:branch'));
  state.failInjection = false;
  state.events.length = 0;
  await runRootMixer(recorder, { ...input, weights: { ...input.weights, inject: null }, mixBranch: true });
  assert.ok(state.events.includes('project:hc_mixer_up') && state.events.includes('fused'));
  assert.ok(!state.events.includes('up-fused'), 'root retains its independently admitted projection/mix path');
  await assert.rejects(runRootMixer(recorder, { ...input, upMixBranch: true }), /root mixer program/);
  await assert.rejects(runGatedResidual(recorder, { ...input, layerIdx: undefined, mixBranch: true }), /explicit layer index/);
} finally { hooks.deregister(); delete globalThis.hcMixBranchWiring; }
// Real ownership classes, no shader/GPU: an error after a submitted prefix
// must settle that prefix without freeing the session's borrowed branch output.
{
  let complete;
  const done = new Promise(resolve => { complete = resolve; });
  globalThis.GPUBufferUsage ??= { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 };
  const device = {
    features: new Set(), lost: new Promise(() => {}),
    queue: { submit() {}, onSubmittedWorkDone: () => done },
    createCommandEncoder: () => ({ finish: () => ({}) }),
    createBuffer: ({ size }) => ({ size, destroyed: false, destroy() { this.destroyed = true; } }),
  };
  setDeviceMemoryBudget(device, 8 * 1024 * 1024);
  const workspace = new PreparedDecodeWorkspace(device), recorder = new CommandRecorder(device);
  recorder.decodeWorkspace = workspace;
  const output = workspace.getBuffer('hc:3:attn_hc:branch', 16);
  recorder.trackTemporaryBuffer(output);
  recorder.submitPartial();
  const retiring = recorder.abort();
  assert.equal(output.destroyed, false, 'borrowed output survives the submitted prefix');
  complete();
  await retiring;
  assert.equal(output.destroyed, false, 'failure cleanup cannot retire a session-owned output');
  assert.equal(workspace.getBuffer('hc:3:attn_hc:branch', 16), output, 'recovery retains the same slot');
  workspace.release();
  assert.equal(output.destroyed, true);
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0, 'explicit owner retirement releases its allocation');
}
console.log('HC mix-branch wiring: opt-in, role-scoped prepared reuse and failure ownership passed');
