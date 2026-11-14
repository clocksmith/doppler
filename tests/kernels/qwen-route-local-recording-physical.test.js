import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer, getBufferPool, isBufferActive } from '../../src/memory/buffer-pool.js';
import { createProfilingRecorder } from '../../src/gpu/command-recorder.js';
import { beginImmediateDispatchProfile } from '../../src/gpu/kernels/dispatch.js';
import { createKernelRegistry, enterKernelRegistry } from '../../src/gpu/kernels/kernel-configs.js';
import { runQwenRouteExperts } from '../../src/inference/pipelines/text/moe/qwen-route-executor.js';
import { packedFixture } from '../helpers/q4_0-fixture.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Local route recording requires physical WebGPU: ${probe.reason}`);
const device = getDevice(), errors = [], owned = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = data => {
  const buffer = acquireBuffer(data.byteLength);
  owned.push(buffer);
  uploadData(buffer, data);
  return buffer;
};
const E = 3, H = 256, I = 256, topK = 2;
const bank = (dtype, shape) => createWeightBuffer(upload(packedFixture(dtype, shape[0] * shape[1], shape[2])), dtype, 'row', shape);
try {
  const input = createTensor(upload(Float32Array.from({ length: H }, (_, i) => Math.sin(i * 0.13) * 0.1)), 'f32', [1, H]);
  const indices = upload(new Uint32Array([2, 0]));
  const routes = upload(new Float32Array([0.3, 0.7]));
  const gate = bank('q4_0', [E, I, H]), up = bank('q4_0', [E, I, H]);
  for (const dtype of ['q4_0', 'q4_1']) {
    const down = bank(dtype, [E, H, I]);
    const options = { inputTensor: input, indicesBuffer: indices, weightsBuffer: routes,
      weights: { numExperts: E, expertFormat: 'qwen3_5', gate, up, down },
      expectedExpertFormat: 'qwen3_5', numTokens: 1, topK, hiddenSize: H, intermediateSize: I,
      activationDtype: 'f32', hiddenActivation: 'silu', swigluLimit: null };
    const expectedTensor = await runQwenRouteExperts(options);
    const expected = new Float32Array(await readBuffer(expectedTensor.buffer, H * 4));
    releaseBuffer(expectedTensor.buffer);
    const baselineActive = getBufferPool().getStats().activeBuffers;
    let submits = 0;
    const submit = device.queue.submit.bind(device.queue);
    device.queue.submit = commands => { submits++; return submit(commands); };
    let output;
    try { output = await runQwenRouteExperts({ ...options, phase: 'decode' }); }
    finally { device.queue.submit = submit; }
    assert.equal(submits, 1, 'One compound operator must submit four passes once');
    const actual = new Float32Array(await readBuffer(output.buffer, H * 4));
    assert.deepEqual(actual, expected, 'Local recording must preserve exact F32 output');
    releaseBuffer(output.buffer);
    await device.queue.onSubmittedWorkDone();
    assert.equal(getBufferPool().getStats().activeBuffers, baselineActive);

    output = null;
    // Borrowing an existing timestamp owner covers the same four passes.
    const profiler = createProfilingRecorder('local-route-coverage', device, { profilePhase: 'decode' });
    const release = beginImmediateDispatchProfile(device, () => profiler);
    try {
      output = await runQwenRouteExperts({ ...options, phase: 'decode' });
      release();
      await profiler.submitAndWait();
      await profiler.resolveProfileTimings({ requireComplete: true });
      assert.equal(profiler.getStats().computePassCount, 4);
      assert.equal(profiler.getStats().profile.unmeasuredPasses, 0);
      assert.deepEqual(new Float32Array(await readBuffer(output.buffer, H * 4)), expected);
    } finally { release(); if (output) releaseBuffer(output.buffer); await profiler.abort(); }

    const leave = enterKernelRegistry(createKernelRegistry({ validators: { scatter_add_routes: { f32: {
      id: 'local-route-failure', validate() { throw new Error('Injected route scatter failure'); },
    } } } }));
    try {
      await assert.rejects(runQwenRouteExperts({ ...options, phase: 'decode' }), /Injected route scatter failure/);
    } finally { leave(); }
    assert.equal(getBufferPool().getStats().activeBuffers, baselineActive);
    assert.ok(isBufferActive(input.buffer) && isBufferActive(indices) && isBufferActive(routes));
    console.log(JSON.stringify({ dtype, exact: true, submits, profiledPasses: 4, failureCleanup: true }));
  }
  assert.deepEqual(errors, []);
} finally {
  await device.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
