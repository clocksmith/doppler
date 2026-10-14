import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/gpu/kernels/paired-linear-attention.js', import.meta.url).href;
const calls = [];
let resolvePipeline, rejectPipeline;
globalThis.pairedPreparation = (...args) => { calls.push(args); return new Promise((resolve, reject) => { resolvePipeline = resolve; rejectPipeline = reject; }); };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === './pipeline-cache.js') return { shortCircuit: true,
    url: 'data:text/javascript,export const getPipelineFast=(...args)=>globalThis.pairedPreparation(...args)' };
  return next(specifier, context);
} });
const { preparePairedLinearAttentionPipelines } = await import(target);
const device = {};
async function flush() { for (let i = 0; i < 6; i++) await Promise.resolve(); }
try {
  await assert.rejects(preparePairedLinearAttentionPipelines(undefined, undefined, 'silu', {}), /owning GPU device/);
  const aborted = new AbortController(); aborted.abort(Error('cancelled before opening'));
  await assert.rejects(preparePairedLinearAttentionPipelines(device, aborted.signal, 'silu', {}), /cancelled before/);
  await assert.rejects(preparePairedLinearAttentionPipelines(device, undefined, undefined, {}), /norm-gate activation/);
  assert.equal(calls.length, 0);
  const stages = [
    ['gated_delta_conv', 'paired', 64],
    ['qwen_linear_attention_prepare', 'paired_beta', 64],
    ['qwen_linear_attention_prepare', 'paired_decay_scale', 64],
    ['qwen_linear_attention_prepare', 'paired_log_decay', 64],
    ['qwen_linear_attention_prepare', 'paired_decay', 64],
    ['gated_delta_recurrent', 'paired_qk', 128],
    ['gated_delta_recurrent', 'paired_state', 128],
    ['gated_delta_recurrent', 'paired_norm', 128],
    ['gated_delta_recurrent', 'paired_gate', 128],
  ];
  const successful = preparePairedLinearAttentionPipelines(device, undefined, 'silu', {}); await flush();
  for (const [i, [operation, variant, width]] of stages.entries()) {
    assert.deepEqual(calls[i], [operation, variant, null, { WORKGROUP_SIZE: width }, device]);
    resolvePipeline(); await flush();
  }
  await successful;
  for (const stage of stages.keys()) {
    const count = calls.length, controller = new AbortController(); let settled = false;
    const opening = preparePairedLinearAttentionPipelines(device, controller.signal, 'silu', {});
    opening.then(() => { settled = true; }, () => { settled = true; });
    for (let i = 0; i < stage; i++) { resolvePipeline(); await flush(); }
    controller.abort(Error('cancelled while compiling')); await flush();
    assert.equal(settled, false, 'a pending provider task stays owned until it settles');
    resolvePipeline(); await assert.rejects(opening, /cancelled while compiling/);
    assert.equal(calls.length, count + stage + 1, 'abort must not start another pipeline');
  }
  const sigmoid = preparePairedLinearAttentionPipelines(device, undefined, 'sigmoid', {}); await flush();
  for (const [operation, variant, width] of stages) {
    assert.deepEqual(calls.at(-1), [operation, variant === 'paired_gate' ? 'paired_gate_sigmoid' : variant, null,
      { WORKGROUP_SIZE: width }, device]);
    resolvePipeline(); await flush();
  }
  await sigmoid;
  const rejected = preparePairedLinearAttentionPipelines(device, undefined, 'silu', {});
  rejectPipeline(Error('provider compile failure')); await assert.rejects(rejected, /provider compile failure/);
  // A failure does not clear another owner cache or permanently cache a rejected preparation at this boundary.
  const nextDevice = {}; const retry = preparePairedLinearAttentionPipelines(nextDevice, undefined, 'silu', {});
  assert.equal(calls.at(-1)[4], nextDevice); for (const _stage of stages) { resolvePipeline(); await flush(); } await retry;
} finally { hooks.deregister(); delete globalThis.pairedPreparation; }
console.log('paired pipeline preparation: exact variants/constants/device, rejection and cancellation settlement passed');
