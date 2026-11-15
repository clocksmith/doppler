import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { deriveChannelRotation, applyChannelRotation } from '../../src/gpu/kernels/channel-rotation.js';
import { rotationReference, applyRotationReference } from './reference/channel-rotation.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice();
const adapter = getKernelCapabilities().adapterInfo;
assert.doesNotMatch(JSON.stringify(adapter), /swiftshader|llvmpipe|software/i);
const owned = new Set();
const keep = tensor => { owned.add(tensor.buffer); return tensor; };
function tensor(values, shape) {
  const data = new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, 'channel-rotation-fixture');
  owned.add(buffer); uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape);
}
const read = async t => new Float32Array(await readBuffer(t.buffer, t.shape.reduce((a, b) => a * b, 1) * 4));
function near(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Number.isFinite(value) && Math.abs(value - expected[i]) <= 2e-5 * Math.max(1, Math.abs(expected[i])), `${i}: ${value} vs ${expected[i]}`));
}
const timings = [];
let recorder = null;
device.pushErrorScope('validation');
try {
  for (const channels of [3, 257, 4096]) {
    const data = Float32Array.from({ length: channels * 4 }, (_, i) => Math.sin(i * 0.37) + Math.cos(i * 0.13));
    const input = tensor(data, [4, channels]);
    const options = { tokens: 4, channels, start: 1, stop: 2, contentStart: 1, contentStop: 3, epsilon: 1e-6 };
    const expected = rotationReference(data, options);
    assert.equal(expected.status, 1);
    for (const transport of ['immediate', 'recorded']) {
      await device.queue.onSubmittedWorkDone();
      const begin = performance.now();
      recorder = transport === 'recorded' ? createCommandRecorder('channel-rotation-physical') : null;
      const rotation = await deriveChannelRotation(recorder, input, options);
      keep(rotation.basis); keep(rotation.status);
      const positive = keep(await applyChannelRotation(recorder, input, rotation, { tokens: 4, channels, theta: 0.05 }));
      const inverse = keep(await applyChannelRotation(recorder, positive, rotation, { tokens: 4, channels, theta: -0.05 }));
      const zero = keep(await applyChannelRotation(recorder, input, rotation, { tokens: 4, channels, theta: 0 }));
      if (recorder) await recorder.submitAndWait();
      await device.queue.onSubmittedWorkDone();
      timings.push({ channels, transport, deriveAndThreeApplicationsCompletedWallMs: performance.now() - begin,
        logicalBasisAndStatusBytes: channels * 8 + 4 });
      recorder = null;
      assert.equal((await read(rotation.status))[0], 1);
      near(await read(rotation.basis), expected.basis);
      near(await read(positive), applyRotationReference(data, expected, { channels, theta: 0.05 }));
      near(await read(inverse), data);
      assert.deepEqual(await read(zero), data);
      assert.deepEqual(await read(input), data);
    }
  }
  for (const data of [[0,0,0,1,2,3], [1,2,3,2,4,6]]) {
    const input = tensor(data, [2, 3]);
    const rotation = await deriveChannelRotation(null, input,
      { tokens: 2, channels: 3, start: 0, stop: 1, contentStart: 0, contentStop: 2, epsilon: 1e-6 });
    keep(rotation.basis); keep(rotation.status);
    const output = keep(await applyChannelRotation(null, input, rotation, { tokens: 2, channels: 3, theta: 0.05 }));
    const status = (await read(rotation.status))[0];
    assert.equal(status, 0);
    assert.deepEqual(await read(output), new Float32Array(data));
    await assert.rejects(deriveChannelRotation(null, input,
      { tokens: 2, channels: 3, start: 0, stop: 2, contentStart: 0, contentStop: 2, epsilon: 1e-6 }), /nonempty complement/);
    await assert.rejects(applyChannelRotation(null, input, rotation, { tokens: 2, channels: 3, theta: Infinity }), /finite f32/);
  }
  const badInput = tensor([NaN, 2, 3, 1, 2, 4], [2, 3]);
  const badOptions = { tokens: 2, channels: 3, start: 0, stop: 1, contentStart: 0, contentStop: 2, epsilon: 1e-6 };
  await assert.rejects(deriveChannelRotation(null, badInput, badOptions), /invalid or overflowing/);
  recorder = createCommandRecorder('channel-rotation-invalid');
  const badRotation = await deriveChannelRotation(recorder, badInput, badOptions);
  keep(badRotation.basis); keep(badRotation.status);
  await assert.rejects(recorder.submitAndWait(), /invalid or overflowing/);
  recorder = null;
  assert.equal(await device.popErrorScope(), null);
  console.log(JSON.stringify({ passed: true, test: 'channel-rotation-physical', adapter, timings,
    limitation: 'Operator-only correctness; synchronized wall times include setup/compilation and are not model overhead or quality evidence.' }));
} finally {
  if (recorder) await recorder.abort();
  await device.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice(); releaseNodeWebGPU();
}
