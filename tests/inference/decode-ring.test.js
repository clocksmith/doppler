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

const { DecodeRing } = await import('../../src/inference/decode-ring.js');
const { setDevice } = await import('../../src/gpu/device.js');

class FakeBuffer {
  constructor({ size, usage, label }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice() {
  const createdBuffers = [];
  return {
    createdBuffers,
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
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
    createBuffer({ size, usage, label }) {
      const buffer = new FakeBuffer({ size, usage, label });
      createdBuffers.push(buffer);
      return buffer;
    },
  };
}

function makeConfig(overrides = {}) {
  return {
    batchSize: 1,
    tokensPerInterval: 4,
    stopCheckMode: 'per-token',
    ringTokens: 2,
    ringStop: 2,
    ringStaging: 2,
    ...overrides,
  };
}

// initial state: uninitialized ring
{
  const ring = new DecodeRing();
  assert.equal(ring.buffers, null);
  assert.equal(ring.config, null);
  assert.equal(ring.index, 0);
  assert.equal(ring.ringSize, 0);
  assert.equal(ring.zeroStopData, null);
  assert.equal(ring.stats, null);
}

// ensure rejects null/undefined config
{
  const ring = new DecodeRing();
  assert.throws(() => ring.ensure(null), /requires config/);
  assert.throws(() => ring.ensure(undefined), /requires config/);
}

// ensure rejects invalid batchSize
{
  const ring = new DecodeRing();
  assert.throws(
    () => ring.ensure({ batchSize: 0, tokensPerInterval: 4, stopCheckMode: 'per-token' }),
    /positive batchSize/
  );
  assert.throws(
    () => ring.ensure({ batchSize: -1, tokensPerInterval: 4, stopCheckMode: 'per-token' }),
    /positive batchSize/
  );
  assert.throws(
    () => ring.ensure({ batchSize: NaN, tokensPerInterval: 4, stopCheckMode: 'per-token' }),
    /positive batchSize/
  );
}

// ensure rejects invalid tokensPerInterval
{
  const ring = new DecodeRing();
  assert.throws(
    () => ring.ensure({ batchSize: 1, tokensPerInterval: 0, stopCheckMode: 'per-token' }),
    /positive tokensPerInterval/
  );
  assert.throws(
    () => ring.ensure({ batchSize: 1, tokensPerInterval: -5, stopCheckMode: 'per-token' }),
    /positive tokensPerInterval/
  );
  assert.throws(
    () => ring.ensure({ batchSize: 1, tokensPerInterval: Infinity, stopCheckMode: 'per-token' }),
    /positive tokensPerInterval/
  );
}

// ensure rejects missing stopCheckMode
{
  const ring = new DecodeRing();
  assert.throws(
    () => ring.ensure({ batchSize: 1, tokensPerInterval: 4 }),
    /requires stopCheckMode/
  );
  assert.throws(
    () => ring.ensure({ batchSize: 1, tokensPerInterval: 4, stopCheckMode: '' }),
    /requires stopCheckMode/
  );
}

// ensure rejects invalid ring sizes (negative, NaN)
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  assert.throws(
    () => ring.ensure(makeConfig({ ringTokens: -1 })),
    /positive ring sizes/
  );
  assert.throws(
    () => ring.ensure(makeConfig({ ringStop: NaN })),
    /positive ring sizes/
  );
  assert.throws(
    () => ring.ensure(makeConfig({ ringStaging: Infinity })),
    /positive ring sizes/
  );
}

// ensure allocates buffers with per-token stop mode
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig());

  assert.ok(ring.buffers);
  assert.equal(ring.buffers.tokens.length, 2);
  assert.equal(ring.buffers.stop.length, 2);
  assert.equal(ring.buffers.stagingTokens.length, 2);
  assert.equal(ring.buffers.stagingStop.length, 2);
  assert.equal(ring.index, 0);
  assert.equal(ring.ringSize, 2);
  assert.ok(ring.zeroStopData instanceof Uint32Array);
  assert.equal(ring.zeroStopData.length, 5);
  assert.ok(ring.stats);

  ring.release();
}

// ensure allocates without stop buffers for non-per-token mode
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ stopCheckMode: 'batch' }));

  assert.ok(ring.buffers);
  assert.equal(ring.buffers.tokens.length, 2);
  assert.equal(ring.buffers.stop, null);
  assert.equal(ring.buffers.stagingTokens.length, 2);
  assert.equal(ring.buffers.stagingStop, null);
  assert.equal(ring.zeroStopData, null);

  ring.release();
}

// ensure with null ring sizes produces null buffer arrays
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ ringTokens: null, ringStop: null, ringStaging: null }));

  assert.ok(ring.buffers);
  assert.equal(ring.buffers.tokens, null);
  assert.equal(ring.buffers.stop, null);
  assert.equal(ring.buffers.stagingTokens, null);
  assert.equal(ring.buffers.stagingStop, null);
  assert.equal(ring.ringSize, 1);

  ring.release();
}

// ensure is idempotent with same config
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  const config = makeConfig();
  ring.ensure(config);

  const firstBuffers = ring.buffers;
  const bufferCountAfterFirst = device.createdBuffers.length;

  ring.ensure(config);
  assert.equal(ring.buffers, firstBuffers);
  assert.equal(device.createdBuffers.length, bufferCountAfterFirst);

  ring.release();
}

// ensure with different config reallocates
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ tokensPerInterval: 4 }));
  const firstBuffers = ring.buffers;

  ring.ensure(makeConfig({ tokensPerInterval: 8 }));
  assert.notEqual(ring.buffers, firstBuffers);

  ring.release();
}

// acquire returns null when uninitialized
{
  const ring = new DecodeRing();
  const slot = ring.acquire();
  assert.equal(slot, null);
}

// acquire returns slot with correct fields
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig());

  const slot = ring.acquire();
  assert.ok(slot);
  assert.equal(slot.index, 0);
  assert.ok(slot.tokens);
  assert.ok(slot.stop);
  assert.ok(slot.stagingTokens);
  assert.ok(slot.stagingStop);
  assert.equal(slot.tokensPerInterval, 4);
  assert.ok(slot.zeroStopData instanceof Uint32Array);

  ring.release();
}

// acquire wraps around ring buffers
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ ringTokens: 3, ringStop: 3, ringStaging: 3 }));

  const slot0 = ring.acquire();
  assert.equal(slot0.index, 0);
  assert.equal(slot0.tokens, ring.buffers.tokens[0]);

  ring.advance();
  const slot1 = ring.acquire();
  assert.equal(slot1.index, 1);
  assert.equal(slot1.tokens, ring.buffers.tokens[1]);

  ring.advance();
  const slot2 = ring.acquire();
  assert.equal(slot2.index, 2);
  assert.equal(slot2.tokens, ring.buffers.tokens[2]);

  ring.advance();
  const slot3 = ring.acquire();
  assert.equal(slot3.index, 0);
  assert.equal(slot3.tokens, ring.buffers.tokens[0]);

  ring.release();
}

// advance is a no-op when uninitialized
{
  const ring = new DecodeRing();
  ring.advance();
  assert.equal(ring.index, 0);
}

// advance wraps index around ringSize
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ ringTokens: 2, ringStop: 2, ringStaging: 2 }));
  assert.equal(ring.ringSize, 2);

  ring.advance();
  assert.equal(ring.index, 1);

  ring.advance();
  assert.equal(ring.index, 0);

  ring.advance();
  assert.equal(ring.index, 1);

  ring.release();
}

// reset restores index to 0 and updates stats
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig());

  ring.advance();
  ring.advance();
  ring.acquire();
  assert.equal(ring.index, 0);

  const statsBefore = ring.getStats();
  assert.equal(statsBefore.advances, 2);
  assert.equal(statsBefore.acquires, 1);
  assert.equal(statsBefore.resets, 0);

  ring.reset();
  assert.equal(ring.index, 0);

  const statsAfter = ring.getStats();
  assert.equal(statsAfter.resets, 1);
  assert.equal(statsAfter.advances, 0);
  assert.equal(statsAfter.acquires, 0);

  ring.release();
}

// getStats returns null when uninitialized
{
  const ring = new DecodeRing();
  assert.equal(ring.getStats(), null);
}

// getStats returns a snapshot (not a reference)
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig());

  const stats1 = ring.getStats();
  ring.acquire();
  ring.advance();
  const stats2 = ring.getStats();

  assert.equal(stats1.acquires, 0);
  assert.equal(stats2.acquires, 1);
  assert.equal(stats1.advances, 0);
  assert.equal(stats2.advances, 1);

  ring.release();
}

// stats track slot uses and reuses
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ ringTokens: 2, ringStop: 2, ringStaging: 2 }));

  ring.acquire();
  ring.advance();
  ring.acquire();
  ring.advance();

  ring.acquire();

  const stats = ring.getStats();
  assert.equal(stats.acquires, 3);
  assert.equal(stats.advances, 2);
  assert.equal(stats.tokens.uses, 3);
  assert.equal(stats.tokens.allocated, 2);
  assert.equal(stats.tokens.reuses, 1);

  ring.release();
}

// release destroys all buffers and resets state
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig());

  const allBuffers = [
    ...ring.buffers.tokens,
    ...ring.buffers.stop,
    ...ring.buffers.stagingTokens,
    ...ring.buffers.stagingStop,
  ];

  ring.release();

  for (const buffer of allBuffers) {
    assert.equal(buffer.destroyed, true);
  }
  assert.equal(ring.buffers, null);
  assert.equal(ring.config, null);
  assert.equal(ring.index, 0);
  assert.equal(ring.ringSize, 0);
  assert.equal(ring.zeroStopData, null);
  assert.equal(ring.stats, null);
}

// release is safe to call on uninitialized ring
{
  const ring = new DecodeRing();
  ring.release();
  assert.equal(ring.buffers, null);
}

// release is safe to call twice
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig());
  ring.release();
  ring.release();
  assert.equal(ring.buffers, null);
}

// ringSize is max of token/stop/staging sizes (asymmetric ring counts)
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ ringTokens: 3, ringStop: 5, ringStaging: 4 }));
  assert.equal(ring.ringSize, 5);

  ring.release();
}

// single-element ring wraps on every advance
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ ringTokens: 1, ringStop: 1, ringStaging: 1 }));
  assert.equal(ring.ringSize, 1);

  const slot0 = ring.acquire();
  ring.advance();
  assert.equal(ring.index, 0);

  const slot1 = ring.acquire();
  assert.equal(slot1.tokens, slot0.tokens);

  ring.release();
}

// fractional config values are floored
{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  ring.ensure(makeConfig({ batchSize: 1.9, tokensPerInterval: 4.7, ringTokens: 2.5 }));

  assert.equal(ring.config.batchSize, 1);
  assert.equal(ring.config.tokensPerInterval, 4);
  assert.equal(ring.config.ringTokens, 2);

  ring.release();
}

// buffer size limits are enforced
{
  const device = createFakeDevice();
  device.limits.maxBufferSize = 8;
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  assert.throws(
    () => ring.ensure(makeConfig({ tokensPerInterval: 100 })),
    /exceeds maxBufferSize/
  );
}

// storage binding size limits are enforced
{
  const device = createFakeDevice();
  device.limits.maxStorageBufferBindingSize = 8;
  setDevice(device, { platformConfig: null });

  const ring = new DecodeRing();
  assert.throws(
    () => ring.ensure(makeConfig({ tokensPerInterval: 100 })),
    /exceeds maxStorageBufferBindingSize/
  );
}

// ensure requires GPU device
{
  setDevice(null);

  const ring = new DecodeRing();
  assert.throws(
    () => ring.ensure(makeConfig()),
    /GPU device not initialized/
  );
}

setDevice(null);
console.log('decode-ring.test: ok');
