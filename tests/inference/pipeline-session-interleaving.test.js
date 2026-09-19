import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createInitializedPipeline } from '../../src/inference/pipelines/factory.js';
import { applyPipelineContexts, restorePipelineContexts } from '../../src/inference/pipelines/context.js';
import { getDevice, setDevice } from '../../src/gpu/device.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';
import {
  createShaderSourceScope, bindStorageShaderSourceScope, getScopedShaderSource,
} from '../../src/gpu/kernels/shader-source-scope.js';
import { getBufferPool, releaseBuffer, isBufferActive } from '../../src/memory/buffer-pool.js';

// Contract doubles exercise the real factory, compatibility scope and pool.
// No model arithmetic or physical GPU execution is claimed by these tests.
function deviceFixture(label) {
  const loss = Promise.withResolvers();
  const device = {
    label,
    lost: loss.promise,
    features: new Set(),
    limits: {
      maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64, maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384, maxStorageBuffersPerShaderStage: 8,
      maxUniformBufferBindingSize: 65536, maxComputeWorkgroupsPerDimension: 65535,
    },
    queue: { submit() {}, async onSubmittedWorkDone() {} },
    createBindGroup(descriptor) { return descriptor; },
    createBuffer({ size, usage }) {
      return { size, usage, destroyed: false, destroy() { this.destroyed = true; } };
    },
  };
  return { device, loss };
}

class LifecyclePipeline {
  async initialize(contexts) {
    this.runtimeConfig = applyPipelineContexts(this, contexts, {
      assignGpuContext: true, assignStorageContext: true,
    }).runtimeConfig;
  }

  observe() {
    return {
      device: getDevice(),
      temperature: getRuntimeConfig().inference.sampling.temperature,
      shader: getScopedShaderSource('session.wgsl').source,
    };
  }

  async loadModel(fixture) {
    this.fixture = fixture;
    this.buffer = getBufferPool(this.gpuContext.device).acquire(16, GPUBufferUsage.STORAGE, fixture.id);
    fixture.instances.push(this);
    fixture.started.resolve(this.observe());
    await fixture.resume.promise;
    fixture.signal?.throwIfAborted();
    fixture.observations.push(this.observe());
  }

  async read() {
    await Promise.resolve();
    assert.equal(isBufferActive(this.buffer), true);
    return this.observe();
  }

  async *stream() {
    try {
      yield await this.read();
      yield await this.read();
    } finally {
      this.fixture.streamCleanup.push(this.observe());
      if (this.fixture.streamFailure) throw this.fixture.streamFailure;
    }
  }

  async unload() {
    this.fixture.unloads += 1;
    try {
      if (!this.buffer.destroyed) releaseBuffer(this.buffer);
      this.fixture.released = !isBufferActive(this.buffer);
      this.buffer = null;
      if (this.fixture.unloadFailure) throw this.fixture.unloadFailure;
    } finally {
      restorePipelineContexts(this);
    }
  }
}

function modelFixture(id, device, temperature) {
  const storage = {};
  bindStorageShaderSourceScope(storage, createShaderSourceScope(new Map([['session.wgsl', id]])));
  return {
    id, started: Promise.withResolvers(), resume: Promise.withResolvers(),
    instances: [], observations: [], streamCleanup: [], unloads: 0,
    contexts: { gpu: { device }, storage, runtimeConfig: { inference: { sampling: { temperature } } } },
    expected: { device, temperature, shader: id },
  };
}

for (const sharedDevice of [true, false]) {
  test(`cancelled load releases its scope before another load (shared device: ${sharedDevice})`,
    { timeout: 5000 }, async () => {
      const originalConfig = getRuntimeConfig();
      const originalDevice = getDevice();
      const deviceA = deviceFixture('A').device;
      const deviceB = sharedDevice ? deviceA : deviceFixture('B').device;
      const a = modelFixture('A', deviceA, 0.25);
      const b = modelFixture('B', deviceB, 0.75);
      const cancel = new AbortController();
      a.signal = cancel.signal;
      a.unloadFailure = new Error('cleanup failure must not mask cancellation');
      const reason = new DOMException('cancel A', 'AbortError');
      let loadedB;
      try {
        const openingA = createInitializedPipeline(LifecyclePipeline, a, a.contexts);
        const rejectedA = assert.rejects(openingA, error => error === reason);
        assert.deepEqual(await a.started.promise, a.expected);
        const openingB = createInitializedPipeline(LifecyclePipeline, b, b.contexts);
        b.resume.resolve();
        await Promise.resolve();
        assert.equal(b.instances.length, 0, 'the legacy factory must not overlap active contexts');
        setRuntimeConfig({ inference: { sampling: { temperature: 1.5 } } });
        cancel.abort(reason);
        a.resume.resolve();
        await rejectedA;
        loadedB = await openingB;
        assert.equal(a.unloads, 1);
        assert.equal(a.released, true, 'A releases its allocation before B can reuse it');
        assert.deepEqual(b.observations, [b.expected]);
        assert.deepEqual(await loadedB.read(), b.expected);
        assert.equal(getRuntimeConfig().inference.sampling.temperature, 1.5);
        assert.equal(getDevice(), originalDevice);
        assert.equal(getScopedShaderSource('session.wgsl'), null);
      } finally {
        a.resume.resolve();
        b.resume.resolve();
        await loadedB?.unload();
        getBufferPool(deviceA).destroy();
        if (!sharedDevice) getBufferPool(deviceB).destroy();
        setRuntimeConfig(originalConfig);
        setDevice(originalDevice);
      }
    });
}

test('stream cleanup failure and loss of A do not invalidate B or its queued work',
  { timeout: 5000 }, async () => {
    const originalConfig = getRuntimeConfig();
    const originalDevice = getDevice();
    const ownerA = deviceFixture('A');
    const ownerB = deviceFixture('B');
    const a = modelFixture('A', ownerA.device, 0.25);
    const b = modelFixture('B', ownerB.device, 0.75);
    a.resume.resolve();
    b.resume.resolve();
    let loadedA, loadedB, streamA, streamB;
    try {
      [loadedA, loadedB] = await Promise.all([
        createInitializedPipeline(LifecyclePipeline, a, a.contexts),
        createInitializedPipeline(LifecyclePipeline, b, b.contexts),
      ]);
      const cleanupFailure = new Error('stream cleanup failed');
      a.streamFailure = cleanupFailure;
      streamA = loadedA.stream();
      assert.deepEqual((await streamA.next()).value, a.expected);
      const queuedB = loadedB.read();
      await assert.rejects(streamA.return(), error => error === cleanupFailure);
      assert.deepEqual(a.streamCleanup, [a.expected]);
      assert.deepEqual(await queuedB, b.expected);

      streamB = loadedB.stream();
      assert.deepEqual((await streamB.next()).value, b.expected);
      const lostRead = assert.rejects(loadedA.read(), /device is lost/);
      const closeA = loadedA.unload();
      ownerA.loss.resolve({ reason: 'destroyed', message: 'lose A while B owns the execution scope' });
      await ownerA.loss.promise;
      assert.equal(getDevice(), ownerB.device);
      assert.equal(isBufferActive(loadedB.buffer), true);
      assert.deepEqual((await streamB.next()).value, b.expected);
      await streamB.return();
      await lostRead;
      await closeA;
      loadedA = null;
      assert.equal(a.unloads, 1);
      assert.deepEqual(await loadedB.read(), b.expected, 'B survives closing A after A loses its device');
      assert.equal(getDevice(), originalDevice);
      assert.equal(getScopedShaderSource('session.wgsl'), null);
    } finally {
      await streamA?.return();
      await streamB?.return();
      await loadedA?.unload();
      await loadedB?.unload();
      getBufferPool(ownerB.device).destroy();
      setRuntimeConfig(originalConfig);
      setDevice(originalDevice);
    }
  });
