import assert from 'node:assert/strict';
import { DecodeBufferManager } from '../../src/inference/decode-buffers.js';
import { setDevice } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { attachPreparedDecodeWorkspace } from '../../src/inference/pipelines/text/execution-plan.js';
import { getBufferDevice } from '../../src/memory/device-residency.js';

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 };
const created = [];
function deviceFixture() {
  let lose;
  const device = {
    features: new Set(), limits: {},
    lost: new Promise(resolve => { lose = resolve; }),
    queue: { onSubmittedWorkDone: async () => {}, submit() {} },
    createBindGroup() { return {}; },
    createBuffer({ size }) {
      const buffer = { size, destroys: 0, destroy() { this.destroys++; } };
      created.push(buffer);
      return buffer;
    },
    createCommandEncoder() { return { finish() { return {}; } }; },
  };
  return { device, lose };
}

const config = { hiddenSize: 256, intermediateSize: 512, activationDtype: 'f32', enablePingPong: true };
const first = deviceFixture(), second = deviceFixture();
const manager = new DecodeBufferManager();
const context = { decodeBuffers: manager, activationDtype: 'f32', config: { useMoE: false } };
try {
  setDevice(first.device, { platformConfig: null });
  assert.throws(() => manager.getWorkspace(first.device), /initialized decode buffers/);
  const buffers = manager.ensureBuffers(config);
  assert.equal(getBufferDevice(buffers.hidden), first.device, 'Dedicated buffers record their actual device');
  assert.equal(manager.ensureBuffers(config), buffers);
  const recorder = new CommandRecorder(first.device, 'prepared_lifecycle', { recordLabels: false, recordDispatches: false });
  const workspace = attachPreparedDecodeWorkspace(true, recorder, context, first.device);
  const scratch = workspace.getBuffer('L0:fixture', 64);
  recorder.trackTemporaryBuffer(scratch);
  await recorder.abort();
  assert.equal(scratch.destroys, 0, 'Cancelled recorder cannot destroy session-owned scratch');
  manager.swapPingPong();
  manager.resetPingPong();
  assert.equal(manager.getHiddenBuffer(), buffers.hidden);
  assert.equal(manager.getWorkspace(first.device), workspace, 'Sequence reset retains compatible scratch');
  assert.equal(workspace.getBuffer('L0:fixture', 64), scratch);

  setDevice(second.device, { platformConfig: null });
  assert.throws(() => manager.ensureBuffers(config), /another GPUDevice/,
    'Matching geometry must not return buffers from the previous device');
  assert.throws(() => manager.ensureBuffers({ ...config, hiddenSize: 512 }), /another GPUDevice/,
    'Changing geometry is not authority to destroy an old device session');
  assert.throws(() => manager.getWorkspace(second.device), /another GPUDevice/);
  assert.equal(buffers.hidden.destroys, 0);
  assert.equal(scratch.destroys, 0);
  setDevice(null, { platformConfig: null });
  assert.throws(() => manager.ensureBuffers(config), /live GPU device/);
  assert.throws(() => manager.getWorkspace(null), /live GPU device/);
  manager.release();
  assert.equal(buffers.hidden.destroys, 1);
  assert.equal(scratch.destroys, 1);
  assert.throws(() => workspace.getBuffer('L0:fixture', 64), /released/);

  setDevice(second.device, { platformConfig: null });
  const replacement = manager.ensureBuffers(config);
  assert.notEqual(replacement.hidden, buffers.hidden);
  const replacementWorkspace = manager.getWorkspace(second.device);
  assert.notEqual(replacementWorkspace, workspace);
  const replacementScratch = replacementWorkspace.getBuffer('L0:fixture', 64);
  second.lose({ reason: 'unknown', message: 'injected device loss' });
  await Promise.resolve();
  assert.throws(() => manager.ensureBuffers(config), /live GPU device/);
  assert.throws(() => manager.getWorkspace(second.device), /live GPU device/);
  assert.throws(() => replacementWorkspace.getBuffer('L0:fixture', 64), /lost/);
  assert.equal(replacementScratch.destroys, 0, 'Device loss preserves the cleanup owner');
  manager.release();
  manager.release();
  assert.equal(replacementScratch.destroys, 1);
  assert.equal(manager.hasBuffers(), false);
  setDevice(first.device, { platformConfig: null });
  const halfBuffers = manager.ensureBuffers({ ...config, activationDtype: 'f16' });
  assert.equal(halfBuffers.hidden.size, 512, 'Declared F16 activations retain their own storage size');
  const beforeRejectedAttachment = created.length;
  assert.throws(() => attachPreparedDecodeWorkspace(true, {}, { ...context, activationDtype: 'f16' }, first.device),
    /Prepared decode workspace requires/, 'Unqualified F16 execution cannot bypass admission');
  assert.equal(created.length, beforeRejectedAttachment, 'Rejected F16 admission allocates no scratch');
} finally {
  manager.release();
  setDevice(null, { platformConfig: null });
}
assert(created.every(buffer => buffer.destroys === 1), 'Each session buffer is destroyed exactly once');
console.log('decode-buffer-device-lifecycle: reuse, cancellation, reset, device replacement and loss passed');
