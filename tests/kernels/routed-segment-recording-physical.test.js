import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runQwenRouteExperts } from '../../src/inference/pipelines/text/moe/qwen-route-executor.js';
import { executeRoutedExpertChunks } from '../../src/inference/pipelines/text/moe/routed-expert-chunks.js';
import { packedFixture } from '../helpers/q4_0-fixture.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Segmented route test needs physical WebGPU: ${probe.reason}`);
const device = getDevice(), owned = [], errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = data => {
  const buffer = acquireBuffer(data.byteLength); owned.push(buffer); uploadData(buffer, data); return buffer;
};
const E = 3, H = 256, I = 256;
const bank = (dtype, shape) => createWeightBuffer(upload(packedFixture(dtype, shape[0] * shape[1], shape[2])), dtype, 'row', shape);
try {
  const input = createTensor(upload(Float32Array.from({ length: H }, (_, i) => Math.sin(i) * 0.1)), 'f32', [1, H]);
  const original = upload(new Uint32Array([3, 1])), indices = upload(new Uint32Array([0, 0]));
  const slots = upload(new Uint32Array([2, 0])), routes = upload(new Float32Array([0.3, 0.7]));
  const map = upload(new Uint32Array([0xffffffff, 0, 0xffffffff, 2]));
  const gate = bank('q4_0', [E, I, H]), up = bank('q4_0', [E, I, H]);
  const policy = { storageMode: 'compact-decode', routeChunkTokens: 256 };
  for (const dtype of ['q4_0', 'q4_1']) {
    const weights = { numExperts: E, expertFormat: 'qwen3_5', gate, up, down: bank(dtype, [E, H, I]) };
    const options = { inputTensor: input, indicesBuffer: slots, weightsBuffer: routes, weights,
      expectedExpertFormat: 'qwen3_5', numTokens: 1, topK: 2, hiddenSize: H, intermediateSize: I,
      activationDtype: 'f32', hiddenActivation: 'silu', swigluLimit: null, layerIdx: 0 };
    const expectedTensor = await runQwenRouteExperts(options);
    const expected = new Float32Array(await readBuffer(expectedTensor.buffer, H * 4)); releaseBuffer(expectedTensor.buffer);
    const before = getBufferPool().getStats().activeBuffers;
    const recorder = createCommandRecorder('segmented-route-test');
    const outputs = [];
    let preparations = 0;
    const loader = {
      getRoutedExpertCapacity() { return E; },
      async prepareRoutedExperts(_layer, selected) {
        assert.deepEqual(selected, [3, 1], 'Unsubmitted routing copy must be flushed before readback');
        if (outputs.length) assert.deepEqual(new Float32Array(await readBuffer(outputs[0].buffer, H * 4)), expected,
          'Previous expert consumer must complete before bank reuse');
        preparations++;
        return { ...weights, sourceExpertToSlot: map, sourceNumExperts: 4 };
      },
    };
    try {
      for (let pass = 0; pass < 2; pass++) {
        recorder.getEncoder().copyBufferToBuffer(original, 0, indices, 0, 8);
        outputs.push(await executeRoutedExpertChunks({ ...options, phase: 'decode', indicesBuffer: indices }, loader, policy, undefined, recorder));
      }
      await recorder.submitAndWait();
      for (const output of outputs) assert.deepEqual(new Float32Array(await readBuffer(output.buffer, H * 4)), expected);
      assert.equal(preparations, 2);
    } finally { await recorder.abort(); for (const output of outputs) releaseBuffer(output.buffer); }
    assert.equal(getBufferPool().getStats().activeBuffers, before);
    const cancelled = createCommandRecorder('segmented-cancel');
    const controller = new AbortController(); controller.abort();
    await assert.rejects(executeRoutedExpertChunks({ ...options, phase: 'decode' }, loader, policy, controller.signal, cancelled), /abort/i);
    await cancelled.abort();
    const failing = createCommandRecorder('segmented-failure');
    failing.getEncoder().copyBufferToBuffer(original, 0, indices, 0, 8);
    await assert.rejects(executeRoutedExpertChunks({ ...options, phase: 'decode', indicesBuffer: indices },
      { ...loader, async prepareRoutedExperts() { throw new Error('Injected prepare failure'); } }, policy, undefined, failing), /Injected/);
    await failing.abort();
    assert.equal(getBufferPool().getStats().activeBuffers, before);
    console.log(JSON.stringify({ dtype, exact: true, routingFlushed: true, previousConsumerBeforeReuse: true, cancellationCleanup: true }));
  }
  assert.deepEqual(errors, []);
} finally {
  await device.queue.onSubmittedWorkDone(); for (const buffer of owned) releaseBuffer(buffer); destroyDevice();
}
