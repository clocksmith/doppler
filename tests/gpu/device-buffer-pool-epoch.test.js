import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

const {
  getDevice,
  getDeviceEpoch,
  setDevice,
} = await import('../../src/gpu/device.js');
const {
  BufferUsage,
  acquireBuffer,
  destroyBufferPool,
  getBufferPool,
  releaseBuffer,
} = await import('../../src/memory/buffer-pool.js');
const { applyPipelineContexts, restorePipelineContexts } = await import('../../src/inference/pipelines/context.js');

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

class FakeBuffer {
  constructor({ size, usage, owner }) {
    this.size = size;
    this.usage = usage;
    this.owner = owner;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice(label, options = {}) {
  const createdBuffers = [];
  const lostDeferred = options.lostDeferred ?? createDeferred();
  const submittedDeferred = options.submittedDeferred ?? null;

  return {
    label,
    createdBuffers,
    lost: lostDeferred.promise,
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return submittedDeferred ? submittedDeferred.promise : Promise.resolve();
      },
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createBuffer({ size, usage }) {
      const buffer = new FakeBuffer({ size, usage, owner: label });
      createdBuffers.push(buffer);
      return buffer;
    },
  };
}

async function flushMicrotasks() {
  await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

{
  destroyBufferPool();
  setDevice(null);
  const device = createFakeDevice('shared-session-device');
  setDevice(device, { platformConfig: null });
  const epoch = getDeviceEpoch();
  const firstSession = {}, secondSession = {};
  applyPipelineContexts(firstSession);
  applyPipelineContexts(secondSession);
  const pool = getBufferPool();
  const first = acquireBuffer(64, BufferUsage.STORAGE, 'first-session');
  const second = acquireBuffer(64, BufferUsage.STORAGE, 'second-session');

  releaseBuffer(first);
  restorePipelineContexts(firstSession);
  const remainingPool = getBufferPool();
  await flushMicrotasks();
  assert.equal(second.destroyed, false, 'restoring one session must not destroy another session\'s live buffer');
  assert.equal(getDeviceEpoch(), epoch, 'rebinding the same physical device preserves its generation');
  assert.equal(remainingPool, pool);
  assert.equal(remainingPool.isActiveBuffer(second), true);

  releaseBuffer(second);
  restorePipelineContexts(secondSession);
  assert.equal(getDeviceEpoch(), epoch);
}

{
  destroyBufferPool();
  setDevice(null);

  const oldLost = createDeferred();
  const currentLost = createDeferred();
  const oldDevice = createFakeDevice('old-null-hop', { lostDeferred: oldLost });
  const currentDevice = createFakeDevice('current-null-hop', { lostDeferred: currentLost });

  setDevice(oldDevice, { platformConfig: null });
  const oldEpoch = getDeviceEpoch();

  setDevice(null);
  const clearedEpoch = getDeviceEpoch();
  assert.equal(getDevice(), null);
  assert.ok(clearedEpoch > oldEpoch);

  setDevice(currentDevice, { platformConfig: null });
  const currentEpoch = getDeviceEpoch();
  assert.ok(currentEpoch > clearedEpoch);

  oldLost.resolve({ message: 'stale device lost after null reset', reason: 'destroyed' });
  await flushMicrotasks();

  assert.equal(getDevice(), currentDevice);
  assert.equal(getDeviceEpoch(), currentEpoch);

  currentLost.resolve({ message: 'current device lost', reason: 'destroyed' });
  await flushMicrotasks();

  assert.equal(getDevice(), null);
  assert.ok(getDeviceEpoch() > currentEpoch);
}

{
  destroyBufferPool();
  setDevice(null);

  const oldLost = createDeferred();
  const currentLost = createDeferred();
  const oldDevice = createFakeDevice('old', { lostDeferred: oldLost });
  const currentDevice = createFakeDevice('current', { lostDeferred: currentLost });

  setDevice(oldDevice, { platformConfig: null });
  const oldEpoch = getDeviceEpoch();
  setDevice(currentDevice, { platformConfig: null });
  const currentEpoch = getDeviceEpoch();

  assert.notEqual(currentEpoch, oldEpoch);

  oldLost.resolve({ message: 'old device lost', reason: 'destroyed' });
  await flushMicrotasks();

  assert.equal(getDevice(), currentDevice);
  assert.equal(getDeviceEpoch(), currentEpoch);

  currentLost.resolve({ message: 'current device lost', reason: 'destroyed' });
  await flushMicrotasks();

  assert.equal(getDevice(), null);
  assert.ok(getDeviceEpoch() > currentEpoch);
}

{
  destroyBufferPool();
  setDevice(null);

  const oldSubmit = createDeferred();
  const oldDevice = createFakeDevice('old', { submittedDeferred: oldSubmit });
  const currentDevice = createFakeDevice('current');

  setDevice(oldDevice, { platformConfig: null });
  const oldPool = getBufferPool();
  const oldBuffer = acquireBuffer(64, BufferUsage.STORAGE, 'epoch_scoped_pool');
  releaseBuffer(oldBuffer);

  setDevice(currentDevice, { platformConfig: null });
  const currentPool = getBufferPool();

  assert.notEqual(currentPool, oldPool);
  assert.equal(oldBuffer.destroyed, false);
  const continued = oldPool.acquire(64, BufferUsage.STORAGE, 'continued_old_session');
  assert.equal(continued.owner, 'old');
  releaseBuffer(continued);
  assert.equal(oldPool.isActiveBuffer(continued), false, 'release routes to the originating pool');

  oldSubmit.resolve();
  await flushMicrotasks();

  assert.equal(oldBuffer.destroyed, false, 'switching the compatibility device must preserve other owners');

  const currentBuffer = acquireBuffer(64, BufferUsage.STORAGE, 'current_pool');
  assert.notEqual(currentBuffer, oldBuffer);
  assert.equal(currentBuffer.owner, 'current');
  destroyBufferPool(oldDevice);
  await flushMicrotasks();
  assert.equal(oldBuffer.destroyed, true);
  assert.equal(currentBuffer.destroyed, false, 'closing A preserves B');
  releaseBuffer(currentBuffer);
}

destroyBufferPool();
setDevice(null);
console.log('device-buffer-pool-epoch.test: ok');
