import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createCommandExecutor } from '../../src/client/runtime/command-executor.js';

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function fixture() {
  const compilation = deferred(), started = deferred(), loss = deferred(), completion = deferred();
  const controller = new AbortController();
  let submits = 0;
  const buffer = { destroyed: false, destroy() { this.destroyed = true; } };
  const binder = { getSlot: () => ({ buffer }) };
  const device = {
    lost: loss.promise,
    createBuffer: descriptor => descriptor,
    createShaderModule: descriptor => descriptor,
    createComputePipelineAsync() { started.resolve(); return compilation.promise; },
    createBindGroup: descriptor => descriptor,
    createCommandEncoder: () => ({
      beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
      finish: () => ({}),
    }),
    queue: { submit() { submits++; }, onSubmittedWorkDone: () => completion.promise },
  };
  const modules = new Map([['probe', { id: 'probe', sourceHash: 'source', source: 'shader', entry: 'main' }]]);
  const command = { kind: 'dispatch', moduleId: 'probe', bindings: [{ binding: 0, slotId: 'input' }], workgroups: [1] };
  const executor = createCommandExecutor(device, binder);
  const run = () => executor.executePhase('prefill', [command], { modules, signal: controller.signal });
  const ready = () => compilation.resolve({ getBindGroupLayout: () => ({}) });
  return { controller, loss, device, command, completion, compilation, started, binder, buffer, run, ready, submits: () => submits };
}

test('abort during compilation rejects without submitting or destroying borrowed buffers', async () => {
  const f = fixture();
  const result = f.run(); await f.started.promise;
  f.controller.abort(); f.ready();
  await assert.rejects(result, error => error.name === 'AbortError' && error.submission === 'not-submitted');
  assert.equal(f.submits(), 0); assert.equal(f.buffer.destroyed, false);
});

test('abort during binding preparation cannot submit', async () => {
  const f = fixture();
  f.binder.getSlot = () => { f.controller.abort(); return { buffer: f.buffer }; };
  f.ready();
  await assert.rejects(f.run(), error => error.name === 'AbortError' && error.submission === 'not-submitted');
  assert.equal(f.submits(), 0);
});

test('device loss during compilation cannot submit', async () => {
  const f = fixture(); const result = f.run(); await f.started.promise;
  f.loss.resolve({ message: 'injected loss' }); await Promise.resolve(); f.ready();
  await assert.rejects(result, error => error.code === 'COMMAND_DEVICE_LOST' && error.submission === 'not-submitted');
  assert.equal(f.submits(), 0);
});

test('post-submission cancellation reports submitted work rather than successful completion', async () => {
  const f = fixture(); f.command.waitForCompletion = true; f.ready();
  const result = f.run();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.submits(), 1);
  f.controller.abort(); f.completion.resolve();
  await assert.rejects(result, error => error.name === 'AbortError' && error.submission === 'submitted');
  assert.equal(f.submits(), 1); assert.equal(f.buffer.destroyed, false);
});

test('successful dispatch distinguishes submission from observed completion', async () => {
  const f = fixture(); f.ready();
  assert.equal((await f.run()).results[0].outcome, 'submitted');
  f.command.waitForCompletion = true; f.completion.resolve();
  assert.equal((await f.run()).results[0].outcome, 'completed');
});

test('failed compilation can be retried', async () => {
  const f = fixture(); f.compilation.reject(new Error('injected compile failure'));
  await assert.rejects(f.run(), /compile failure/);
  f.device.createComputePipelineAsync = async () => ({ getBindGroupLayout: () => ({}) });
  await f.run(); assert.equal(f.submits(), 1);
});
