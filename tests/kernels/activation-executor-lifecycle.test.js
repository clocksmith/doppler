import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runGeLU, recordGeLU } from '../../src/gpu/kernels/gelu.js';
import { runReLU, recordReLU } from '../../src/gpu/kernels/relu.js';
import { clearPipelineCaches } from '../../src/gpu/kernels/pipeline-cache.js';
import { createKernelRegistry, enterKernelRegistry, setKernelValidator } from '../../src/gpu/kernels/kernel-configs.js';
import { acquireBuffer, releaseBuffer, readBuffer, getBufferPool, isBufferActive } from '../../src/memory/buffer-pool.js';

const probe = await probeNodeGPU();
assert.ok(probe.ready, `Physical activation lifecycle test requires WebGPU: ${probe.reason}`);
const device = getDevice();
const data = Float32Array.from([-3, -1, -0.1, 0, 0.1, 1, 3, 5]);
const inputBuffer = acquireBuffer(data.byteLength);
const borrowed = acquireBuffer(data.byteLength);
device.queue.writeBuffer(inputBuffer, 0, data);
const input = createTensor(inputBuffer, 'f32', [data.length], 'activation-lifecycle');
const active = () => getBufferPool().getStats().activeBuffers;
try {
  for (const [name, run, record] of [['gelu', runGeLU, recordGeLU], ['relu', runReLU, recordReLU]]) {
    const immediate = await run(input);
    const recorder = new CommandRecorder(device);
    const recorded = await record(recorder, input, { outputBuffer: borrowed });
    assert.equal(recorded.buffer, borrowed);
    await recorder.submitAndWait();
    const a = new Float32Array(await readBuffer(immediate.buffer, data.byteLength));
    const b = new Float32Array(await readBuffer(recorded.buffer, data.byteLength));
    assert.deepEqual(a, b);
    if (name === 'relu') assert.deepEqual(a, Float32Array.from(data, x => Math.max(0, x)));
    releaseBuffer(immediate.buffer);
    const before = active();
    const original = device.createBindGroup;
    device.createBindGroup = () => { throw new Error('injected binding failure'); };
    try {
      await assert.rejects(run(input), /injected binding failure/);
      assert.equal(active(), before);
      await assert.rejects(run(input, { outputBuffer: borrowed }), /injected binding failure/);
      assert.equal(isBufferActive(borrowed), true);
      const failed = new CommandRecorder(device);
      await assert.rejects(record(failed, input), /injected binding failure/);
      assert.ok(active() > before, 'recorded owned output remains retained until abort');
      await failed.abort();
      await failed.abort();
      assert.equal(active(), before);
    } finally { device.createBindGroup = original; }
    const control = new AbortController(); control.abort(new Error('cancel before allocation'));
    await assert.rejects(run(input, { signal: control.signal }), /cancel before allocation/);
    assert.equal(active(), before);
  }
  clearPipelineCaches();
  const compile = device.createComputePipelineAsync;
  const control = new AbortController();
  device.createComputePipelineAsync = async function(...args) {
    const result = await compile.apply(this, args);
    control.abort(new Error('cancel during compilation'));
    return result;
  };
  const before = active();
  try {
    await assert.rejects(runGeLU(input, { signal: control.signal }), /cancel during compilation/);
    assert.equal(active(), before);
    assert.equal(isBufferActive(inputBuffer), true);
  } finally { device.createComputePipelineAsync = compile; }

  clearPipelineCaches();
  const calls = [];
  let rejectA = true;
  const registryA = createKernelRegistry({ validators: { gelu: { gelu: {
    id: 'test.gelu.a/v1', validate: (context) => {
      calls.push(`A:${context.uniforms.size}`);
      if (rejectA) throw new Error('validator A rejected');
    },
  } } } });
  const registryB = createKernelRegistry({ validators: { gelu: { gelu: {
    id: 'test.gelu.b/v1', validate: (context) => {
      calls.push(`B:${context.uniforms.size}`);
      throw new Error('validator B rejected');
    },
  } } } });
  const baseline = active();
  let bindingCalls = 0;
  const createBindGroup = device.createBindGroup;
  device.createBindGroup = function(...args) {
    bindingCalls++;
    return createBindGroup.apply(this, args);
  };
  try {
    const restoreA = enterKernelRegistry(registryA);
    try {
      await assert.rejects(runGeLU(input), /validator A rejected/); // cold pipeline
      assert.equal(active(), baseline, 'cold rejection releases owned output');
      await assert.rejects(runGeLU(input, { outputBuffer: borrowed }), /validator A rejected/);
      assert.equal(isBufferActive(borrowed), true);
      assert.equal(bindingCalls, 0, 'rejected immediate calls cannot create dispatch bindings');

      rejectA = false;
      const warmed = await runGeLU(input);
      releaseBuffer(warmed.buffer);
      const afterWarm = bindingCalls;
      rejectA = true;
      await assert.rejects(runGeLU(input), /validator A rejected/); // cached pipeline
      assert.equal(active(), baseline, 'cached rejection releases owned output');
      assert.equal(bindingCalls, afterWarm, 'cached rejection cannot dispatch');

      const rejectedRecorder = new CommandRecorder(device);
      await assert.rejects(recordGeLU(rejectedRecorder, input), /validator A rejected/);
      assert.equal(rejectedRecorder.getStats().dispatches.length, 0);
      assert.ok(active() > baseline, 'recorded owned output is retained until recorder cleanup');
      await rejectedRecorder.abort();
      assert.equal(active(), baseline);
      const borrowedRecorder = new CommandRecorder(device);
      await assert.rejects(recordGeLU(borrowedRecorder, input, { outputBuffer: borrowed }), /validator A rejected/);
      await borrowedRecorder.abort();
      assert.equal(isBufferActive(borrowed), true);
      assert.equal(bindingCalls, afterWarm);
    } finally { restoreA(); }

    setKernelValidator('gelu', 'gelu', () => { throw new Error('late compatibility validator'); });
    const restoreB = enterKernelRegistry(registryB);
    try {
      await assert.rejects(runGeLU(input), /validator B rejected/);
      assert.equal(active(), baseline);
    } finally { restoreB(); }
    const dispatchedSize = inputBuffer.size / Float32Array.BYTES_PER_ELEMENT;
    assert.deepEqual(calls, [
      `A:${dispatchedSize}`, `A:${dispatchedSize}`, `A:${dispatchedSize}`,
      `A:${dispatchedSize}`, `A:${dispatchedSize}`, `A:${dispatchedSize}`, `B:${dispatchedSize}`,
    ]);
  } finally { device.createBindGroup = createBindGroup; }
  console.log(JSON.stringify({ test: 'activation-executor-lifecycle', passed: true,
    adapter: getKernelCapabilities().adapterInfo, scope: 'operator parity and ownership; not model qualification' }));
} finally {
  await device.queue.onSubmittedWorkDone();
  releaseBuffer(inputBuffer); releaseBuffer(borrowed);
}
