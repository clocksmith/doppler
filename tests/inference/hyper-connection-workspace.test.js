import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';

const target = new URL('../../src/inference/pipelines/text/hyper-connection/decoder-layer.js', import.meta.url).href;
const state = { events: [], released: [], fail: null, before: null };
const tensor = (name, rows, columns, buffer = { size: rows * columns * 4, label: name }) =>
  ({ dtype: 'f32', shape: [rows, columns], buffer });
const call = async (stage, input, options, rows, columns) => {
  const output = tensor(stage, rows, columns, options.outputBuffer);
  state.events.push({ stage, input, output, options });
  await state.before?.(stage);
  if (state.fail === stage) throw new Error(`failed ${stage}`);
  return output;
};
globalThis.hcWorkspaceTest = { state, tensor, call };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL !== target) return next(specifier, context);
  let source;
  if (specifier.endsWith('/buffer-pool.js')) {
    source = 'export const releaseBuffer=b=>globalThis.hcWorkspaceTest.state.released.push(b);';
  } else if (specifier.endsWith('/hyper-connection.js')) {
    source = `const {call:c}=globalThis.hcWorkspaceTest;
      export const recordHyperConnectionNorm=(_r,x,_w,o)=>c('normed',x,o,o.numTokens,o.streamCount*o.streamWidth);
      export const runHyperConnectionNorm=(x,w,o)=>recordHyperConnectionNorm(null,x,w,o);
      export const recordHyperConnectionActivation=(_r,x,o)=>c(o.mode,x,o,o.rows,o.columns);
      export const runHyperConnectionActivation=(x,o)=>recordHyperConnectionActivation(null,x,o);
      export const recordHyperConnectionBranch=(_r,x,_n,o)=>c('branch',x,o,o.numTokens,o.streamWidth);
      export const runHyperConnectionBranch=(x,n,o)=>recordHyperConnectionBranch(null,x,n,o);
      export const recordHyperConnectionMixBranch=recordHyperConnectionBranch,runHyperConnectionMixBranch=runHyperConnectionBranch;
      export const recordHyperConnectionInject=(_r,x,_b,_g,o)=>c('injected',x,o,o.numTokens,o.streamCount*o.streamWidth);
      export const runHyperConnectionInject=(x,b,g,o)=>recordHyperConnectionInject(null,x,b,g,o);
      export const recordHyperConnectionReplicate=(_r,x,o)=>c('replicate',x,o,o.numTokens,o.streamCount*o.streamWidth);
      export const runHyperConnectionReplicate=(x,o)=>recordHyperConnectionReplicate(null,x,o);`;
  } else if (specifier.endsWith('/hyper-connection-up.js')) {
    source = `const {call:c}=globalThis.hcWorkspaceTest;
      export const recordHyperConnectionUpMixBranch=(_r,x,_w,_n,o)=>c('up_branch',x,o,o.numTokens,o.streamWidth);
      export const runHyperConnectionUpMixBranch=(x,w,n,o)=>recordHyperConnectionUpMixBranch(null,x,w,n,o);`;
  }
  return source ? { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true }
    : next(specifier, context);
} });

globalThis.GPUBufferUsage ??= { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 };
let complete = () => {};
let fence = Promise.resolve();
const device = { features: new Set(), lost: new Promise(() => {}),
  queue: { submit() {}, onSubmittedWorkDone: () => fence },
  createCommandEncoder: () => ({ finish: () => ({}) }),
  createBuffer: ({ size, label }) => ({ size, label, destroyed: false, destroy() { this.destroyed = true; } }) };
setDeviceMemoryBudget(device, 8 * 1024 * 1024);
const workspace = new PreparedDecodeWorkspace(device);
const recorder = new CommandRecorder(device);
recorder.decodeWorkspace = workspace;
const input = { streams: tensor('streams', 1, 16),
  weights: { norm: tensor('weight', 1, 16), down: {}, up: {}, inject: {} },
  hyper: { streamCount: 4, streamWidth: 4, projectionRank: 2, normEps: 1e-6, normWeightOffset: false },
  numTokens: 1, layerIdx: 3, role: 'attn_hc',
  project: async (role, x, _weight, columns) => tensor(role, x.shape[0], columns) };
try {
  const { runGatedResidual, runRootMixer, runHyperConnectionDecoderLayer } = await import(target);
  const selected = stage => state.events.findLast(event => event.stage === stage).output.buffer;
  await runGatedResidual(recorder, input);
  const first = ['normed', 'low_rank_silu', 'inject_gate'].map(selected);
  assert.equal(workspace.getStats().buffers, 3);
  assert.equal(workspace.getStats().bytes, (16 + 2 + 4) * 4);
  assert.ok(first.every(buffer => workspace.ownsBuffer(buffer)));
  assert.ok(!workspace.ownsBuffer(selected('branch')), 'unfused branch ownership remains pooled');
  await runGatedResidual(recorder, input);
  assert.ok(['normed', 'low_rank_silu', 'inject_gate'].map(selected).every((buffer, i) => buffer === first[i]),
    'serial tokens reuse stable inputs');
  for (const variant of [{ role: 'mlp_hc' }, { layerIdx: 4 }]) {
    await runGatedResidual(recorder, { ...input, ...variant });
    assert.ok(['normed', 'low_rank_silu', 'inject_gate'].map(selected).every((buffer, i) => buffer !== first[i]));
  }
  for (const flags of [{}, { mixBranch: true }, { upMixBranch: true }]) {
    for (const layerIdx of [undefined, -1, 0.5]) {
      const before = workspace.getStats().buffers;
      await assert.rejects(runGatedResidual(recorder, { ...input, ...flags, layerIdx }), /explicit layer index/);
      assert.equal(workspace.getStats().buffers, before, 'invalid layer fails before allocation');
    }
  }
  const count = workspace.getStats().buffers;
  for (const [owner, tokens] of [[recorder, 2], [null, 1], [{ beginComputePass() {}, trackTemporaryBuffer() {} }, 1]]) {
    state.events.length = 0;
    await runGatedResidual(owner, { ...input, numTokens: tokens, streams: tensor('other', tokens, 16), layerIdx: undefined });
    assert.ok(state.events.every(event => !event.options.outputBuffer), 'unprepared and multi-token paths retain pooled outputs');
  }
  state.events.length = 0;
  await runRootMixer(recorder, { ...input, layerIdx: undefined, weights: { ...input.weights, inject: null } });
  assert.ok(state.events.every(event => !event.options.outputBuffer), 'root mixer gains no intermediate workspace outputs');
  assert.equal(workspace.getStats().buffers, count);
  state.events.length = 0;
  await runRootMixer(recorder, { ...input, layerIdx: 0, mixBranch: true, weights: { ...input.weights, inject: null } });
  assert.equal(workspace.getStats().buffers, count + 1, 'root retains only its previously supported fused branch slot');
  assert.ok(state.events.filter(event => event.stage !== 'branch').every(event => !event.options.outputBuffer));

  state.events.length = 0;
  const branch = async x => tensor('branch_result', x.shape[0], 4);
  await runHyperConnectionDecoderLayer(recorder, { ...input, weights: { attention: input.weights, mlp: input.weights },
    attentionBranch: branch, mlpBranch: branch });
  assert.equal(state.events.filter(event => event.stage === 'injected').length, 2);
  assert.ok(state.events.filter(event => event.stage === 'injected').every(event => !event.options.outputBuffer),
    'residual injection remains pooled');

  // A submitted prefix is still consuming the borrowed slots when a later
  // operator fails. Abort must wait without releasing their session owner.
  fence = new Promise(resolve => { complete = resolve; });
  let prefix = false;
  state.before = async stage => { if (stage === 'low_rank_silu' && !prefix) { prefix = true; recorder.submitPartial(); } };
  state.fail = 'inject_gate';
  await assert.rejects(runGatedResidual(recorder, { ...input, mixBranch: true }), /failed inject_gate/);
  const borrowed = [...first, workspace.getBuffer('hc:3:attn_hc:branch', 16)];
  let retired = false;
  const aborting = recorder.abort().then(() => { retired = true; });
  await Promise.resolve();
  assert.equal(retired, false, 'cleanup waits for the submitted prefix');
  assert.ok(borrowed.every(buffer => !buffer.destroyed));
  complete();
  await aborting;
  assert.ok(borrowed.every(buffer => !buffer.destroyed), 'failure cannot destroy borrowed workspace outputs');
  assert.ok(state.released.every(buffer => !workspace.ownsBuffer(buffer)), 'no borrowed slot reaches pooled release');
  state.before = null; state.fail = null;
  const recovering = new CommandRecorder(device); recovering.decodeWorkspace = workspace;
  await runGatedResidual(recovering, input);
  assert.ok(['normed', 'low_rank_silu', 'inject_gate'].map(selected).every((buffer, i) => buffer === first[i]));
  await recovering.abort();
  workspace.release();
  assert.ok(borrowed.every(buffer => buffer.destroyed));
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
} finally {
  complete();
  await recorder.abort();
  workspace.release();
  hooks.deregister(); delete globalThis.hcWorkspaceTest;
}
console.log('HC workspace: stage/role/layer reuse, root/prefill isolation, prefix failure and retirement passed');
