import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';

global.GPUBufferUsage = {
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

global.GPUMapMode = {
  READ: 0x0001,
  WRITE: 0x0002,
};

const createMockBuffer = (size, usage, label = '') => ({
  size,
  usage,
  label,
  destroy: vi.fn(),
  mapAsync: vi.fn().mockResolvedValue(undefined),
  getMappedRange: vi.fn(() => new ArrayBuffer(size)),
  unmap: vi.fn(),
});

const createMockDevice = () => {
  const device = {
    createBuffer: vi.fn((descriptor) => createMockBuffer(descriptor.size, descriptor.usage, descriptor.label)),
    createCommandEncoder: vi.fn(() => ({
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    queue: {
      submit: vi.fn(),
      writeBuffer: vi.fn(),
      onSubmittedWorkDone: vi.fn().mockResolvedValue(undefined),
    },
    limits: {
      maxBufferSize: 2147483647,
      maxStorageBufferBindingSize: 2147483647,
    },
  };
  return device;
};

vi.mock('../../src/gpu/device.js', () => ({
  getDevice: vi.fn(),
  getDeviceLimits: vi.fn(() => ({
    maxBufferSize: 2147483647,
    maxStorageBufferBindingSize: 2147483647,
  })),
  hasFeature: vi.fn(() => false),
  FEATURES: {
    SHADER_F16: 'shader-f16',
    SUBGROUPS: 'subgroups',
    SUBGROUPS_F16: 'subgroups-f16',
    TIMESTAMP_QUERY: 'timestamp-query',
  },
}));

vi.mock('../../src/gpu/perf-guards.js', () => ({
  allowReadback: vi.fn(() => true),
  trackAllocation: vi.fn(),
}));

vi.mock('../../src/debug/index.js', () => ({
  log: {
    warn: vi.fn(),
    debug: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
  },
  trace: {
    buffers: vi.fn(),
  },
}));

vi.mock('../../src/config/runtime.js', () => ({
  getRuntimeConfig: vi.fn(() => ({
    bufferPool: {
      limits: {
        maxBuffersPerBucket: 8,
        maxTotalPooledBuffers: 64,
      },
      alignment: {
        alignmentBytes: 256,
      },
      bucket: {
        minBucketSizeBytes: 1024,
        largeBufferThresholdBytes: 64 * 1024 * 1024,
        largeBufferStepBytes: 64 * 1024 * 1024,
      },
    },
    memory: {
      heapTesting: {
        heapTestSizes: [8 * 1024 * 1024 * 1024],
        fallbackMaxHeapBytes: 2 * 1024 * 1024 * 1024,
      },
      segmentTesting: {
        segmentTestSizes: [2 * 1024 * 1024 * 1024],
        safeSegmentSizeBytes: 512 * 1024 * 1024,
      },
      addressSpace: {
        targetAddressSpaceBytes: 16 * 1024 * 1024 * 1024,
      },
    },
  })),
}));

describe('gpu/buffer-pool', () => {
  let mockDevice;

  beforeEach(async () => {
    vi.resetModules();
    mockDevice = createMockDevice();
    const { getDevice, getDeviceLimits } = await import('../../src/gpu/device.js');
    getDevice.mockReturnValue(mockDevice);
    getDeviceLimits.mockReturnValue({
      maxBufferSize: 2147483647,
      maxStorageBufferBindingSize: 2147483647,
    });
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('BufferPool class', () => {
    describe('acquire', () => {
      it('creates new buffer when pool is empty', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.acquire(1024);

        expect(mockDevice.createBuffer).toHaveBeenCalledWith(
          expect.objectContaining({
            size: 1024,
            usage: expect.any(Number),
          })
        );
        expect(buffer).toBeDefined();
        expect(buffer.size).toBe(1024);
      });

      it('throws when device not initialized', async () => {
        const { getDevice } = await import('../../src/gpu/device.js');
        getDevice.mockReturnValue(null);

        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        expect(() => pool.acquire(1024)).toThrow('Device not initialized');
      });

      it('reuses pooled buffer of same size and usage', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        const buffer1 = pool.acquire(1024, usage);
        pool.release(buffer1);

        mockDevice.createBuffer.mockClear();
        const buffer2 = pool.acquire(1024, usage);

        expect(mockDevice.createBuffer).not.toHaveBeenCalled();
        expect(buffer2).toBe(buffer1);
      });

      it('uses size bucketing for efficient pooling', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer1 = pool.acquire(500);
        expect(buffer1.size).toBe(1024);

        const buffer2 = pool.acquire(1025);
        expect(buffer2.size).toBe(2048);
      });

      it('aligns buffer size to minimum alignment', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.acquire(100);
        expect(buffer.size).toBe(1024);
      });

      it('applies custom label to buffer', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        pool.acquire(1024, GPUBufferUsage.STORAGE, 'test_buffer');

        expect(mockDevice.createBuffer).toHaveBeenCalledWith(
          expect.objectContaining({
            label: expect.stringContaining('test_buffer'),
          })
        );
      });

      it('throws for buffer exceeding device limits', async () => {
        const { getDeviceLimits } = await import('../../src/gpu/device.js');
        getDeviceLimits.mockReturnValue({
          maxBufferSize: 1000,
          maxStorageBufferBindingSize: 1000,
        });

        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        expect(() => pool.acquire(2000, GPUBufferUsage.STORAGE)).toThrow(/exceeds device/i);
      });
    });

    describe('release', () => {
      it('returns buffer to pool for reuse', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.acquire(1024);
        const stats1 = pool.getStats();
        expect(stats1.activeBuffers).toBe(1);
        expect(stats1.pooledBuffers).toBe(0);

        pool.release(buffer);
        const stats2 = pool.getStats();
        expect(stats2.activeBuffers).toBe(0);
        expect(stats2.pooledBuffers).toBe(1);
      });

      it('warns when releasing untracked buffer', async () => {
        const { log } = await import('../../src/debug/index.js');
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const fakeBuffer = createMockBuffer(1024, GPUBufferUsage.STORAGE);
        pool.release(fakeBuffer);

        expect(log.warn).toHaveBeenCalledWith('BufferPool', expect.stringContaining('not tracked'));
      });

      it('destroys buffer when pool is full', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const config = {
          limits: { maxBuffersPerBucket: 2, maxTotalPooledBuffers: 4 },
          alignment: { alignmentBytes: 256 },
          bucket: {
            minBucketSizeBytes: 1024,
            largeBufferThresholdBytes: 64 * 1024 * 1024,
            largeBufferStepBytes: 64 * 1024 * 1024,
          },
        };
        const pool = new BufferPool(false, config);

        const buffer1 = pool.acquire(1024, GPUBufferUsage.STORAGE);
        const buffer2 = pool.acquire(1024, GPUBufferUsage.STORAGE);
        const buffer3 = pool.acquire(1024, GPUBufferUsage.STORAGE);

        pool.release(buffer1);
        pool.release(buffer2);

        const statsBefore = pool.getStats();
        expect(statsBefore.pooledBuffers).toBe(2);

        pool.release(buffer3);

        expect(mockDevice.queue.onSubmittedWorkDone).toHaveBeenCalled();
      });
    });

    describe('getStats', () => {
      it('tracks allocation count', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        pool.acquire(1024);
        pool.acquire(2048);
        pool.acquire(4096);

        const stats = pool.getStats();
        expect(stats.allocations).toBe(3);
      });

      it('tracks reuse count', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.acquire(1024);
        pool.release(buffer);
        pool.acquire(1024);

        const stats = pool.getStats();
        expect(stats.reuses).toBe(1);
      });

      it('calculates hit rate', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.acquire(1024);
        pool.release(buffer);
        pool.acquire(1024);

        const stats = pool.getStats();
        expect(stats.hitRate).toBe('50.0%');
      });

      it('tracks peak memory', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buf1 = pool.acquire(1024);
        const buf2 = pool.acquire(2048);

        const stats1 = pool.getStats();
        expect(stats1.peakBytesAllocated).toBeGreaterThanOrEqual(3072);

        pool.release(buf1);
        pool.release(buf2);

        const stats2 = pool.getStats();
        expect(stats2.peakBytesAllocated).toBe(stats1.peakBytesAllocated);
      });

      it('tracks active and pooled buffer counts', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buf1 = pool.acquire(1024);
        const buf2 = pool.acquire(2048);

        let stats = pool.getStats();
        expect(stats.activeBuffers).toBe(2);
        expect(stats.pooledBuffers).toBe(0);

        pool.release(buf1);

        stats = pool.getStats();
        expect(stats.activeBuffers).toBe(1);
        expect(stats.pooledBuffers).toBe(1);
      });
    });

    describe('createStagingBuffer', () => {
      it('creates buffer with staging label', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        pool.createStagingBuffer(1024);

        expect(mockDevice.createBuffer).toHaveBeenCalledWith(
          expect.objectContaining({
            label: expect.stringContaining('staging_read'),
          })
        );
      });
    });

    describe('createUploadBuffer', () => {
      it('creates buffer with staging write label', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        pool.createUploadBuffer(1024);

        expect(mockDevice.createBuffer).toHaveBeenCalledWith(
          expect.objectContaining({
            label: expect.stringContaining('staging_write'),
          })
        );
      });
    });

    describe('createUniformBuffer', () => {
      it('creates buffer with uniform label', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        pool.createUniformBuffer(64);

        expect(mockDevice.createBuffer).toHaveBeenCalledWith(
          expect.objectContaining({
            label: expect.stringContaining('uniform'),
          })
        );
      });

      it('aligns uniform buffers to 256 bytes', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.createUniformBuffer(100);

        expect(buffer.size % 256).toBe(0);
      });
    });

    describe('uploadData', () => {
      it('writes data to buffer via queue', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.acquire(1024);
        const data = new Float32Array([1, 2, 3, 4]);
        pool.uploadData(buffer, data);

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalledWith(buffer, 0, data);
      });

      it('supports offset parameter', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.acquire(1024);
        const data = new Float32Array([1, 2, 3, 4]);
        pool.uploadData(buffer, data, 256);

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalledWith(buffer, 256, data);
      });

      it('throws when device not initialized', async () => {
        const { getDevice } = await import('../../src/gpu/device.js');
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buffer = pool.acquire(1024);
        getDevice.mockReturnValue(null);

        expect(() => pool.uploadData(buffer, new Float32Array([1]))).toThrow('Device not initialized');
      });
    });

    describe('clearPool', () => {
      it('destroys all pooled buffers', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buf1 = pool.acquire(1024);
        const buf2 = pool.acquire(2048);
        pool.release(buf1);
        pool.release(buf2);

        const statsBefore = pool.getStats();
        expect(statsBefore.pooledBuffers).toBe(2);

        pool.clearPool();

        const statsAfter = pool.getStats();
        expect(statsAfter.pooledBuffers).toBe(0);
        expect(buf1.destroy).toHaveBeenCalled();
        expect(buf2.destroy).toHaveBeenCalled();
      });
    });

    describe('destroy', () => {
      it('destroys all buffers (active and pooled)', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        const buf1 = pool.acquire(1024);
        const buf2 = pool.acquire(2048);
        const buf3 = pool.acquire(4096);
        pool.release(buf2);

        pool.destroy();

        expect(buf1.destroy).toHaveBeenCalled();
        expect(buf2.destroy).toHaveBeenCalled();
        expect(buf3.destroy).toHaveBeenCalled();

        const stats = pool.getStats();
        expect(stats.activeBuffers).toBe(0);
        expect(stats.pooledBuffers).toBe(0);
      });
    });

    describe('configure', () => {
      it('updates pool configuration', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool();

        pool.configure({ enablePooling: false });

        const buf = pool.acquire(1024);
        pool.release(buf);

        const stats = pool.getStats();
        expect(stats.pooledBuffers).toBe(0);
      });
    });

    describe('debug mode', () => {
      it('tracks buffer metadata in debug mode', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool(true);

        pool.acquire(1024, GPUBufferUsage.STORAGE, 'debug_test');

        const stats = pool.getStats();
        expect(stats.activeBuffers).toBe(1);
      });

      it('detects leaked buffers', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool(true);

        pool.acquire(1024, GPUBufferUsage.STORAGE, 'leaked_buffer');

        await new Promise((r) => setTimeout(r, 10));
        const leaks = pool.detectLeaks(5);
        expect(leaks.length).toBe(1);
        expect(leaks[0].label).toBe('leaked_buffer');
      });

      it('returns empty leaks array when not in debug mode', async () => {
        const { BufferPool } = await import('../../src/gpu/buffer-pool.js');
        const pool = new BufferPool(false);

        pool.acquire(1024);

        const leaks = pool.detectLeaks(0);
        expect(leaks).toEqual([]);
      });
    });
  });

  describe('global buffer pool', () => {
    it('getBufferPool returns singleton', async () => {
      const { getBufferPool, destroyBufferPool } = await import('../../src/gpu/buffer-pool.js');

      const pool1 = getBufferPool();
      const pool2 = getBufferPool();

      expect(pool1).toBe(pool2);

      destroyBufferPool();
    });

    it('destroyBufferPool cleans up global pool', async () => {
      const { getBufferPool, destroyBufferPool } = await import('../../src/gpu/buffer-pool.js');

      const pool = getBufferPool();
      pool.acquire(1024);

      destroyBufferPool();

      const newPool = getBufferPool();
      expect(newPool.getStats().allocations).toBe(0);

      destroyBufferPool();
    });

    it('createBufferPool creates independent instance', async () => {
      const { createBufferPool, getBufferPool, destroyBufferPool } = await import('../../src/gpu/buffer-pool.js');

      const globalPool = getBufferPool();
      const customPool = createBufferPool(true);

      expect(customPool).not.toBe(globalPool);

      customPool.destroy();
      destroyBufferPool();
    });
  });

  describe('convenience functions', () => {
    it('acquireBuffer uses global pool', async () => {
      const { acquireBuffer, getBufferPool, destroyBufferPool } = await import('../../src/gpu/buffer-pool.js');

      acquireBuffer(1024);

      const pool = getBufferPool();
      expect(pool.getStats().activeBuffers).toBe(1);

      destroyBufferPool();
    });

    it('releaseBuffer uses global pool', async () => {
      const { acquireBuffer, releaseBuffer, getBufferPool, destroyBufferPool } = await import('../../src/gpu/buffer-pool.js');

      const buffer = acquireBuffer(1024);
      releaseBuffer(buffer);

      const pool = getBufferPool();
      expect(pool.getStats().pooledBuffers).toBe(1);

      destroyBufferPool();
    });
  });

  describe('withBuffer helper', () => {
    it('automatically releases buffer after use', async () => {
      const { withBuffer, getBufferPool, destroyBufferPool } = await import('../../src/gpu/buffer-pool.js');

      await withBuffer(1024, GPUBufferUsage.STORAGE, async (buffer) => {
        const pool = getBufferPool();
        expect(pool.getStats().activeBuffers).toBe(1);
        return buffer.size;
      });

      const pool = getBufferPool();
      expect(pool.getStats().activeBuffers).toBe(0);
      expect(pool.getStats().pooledBuffers).toBe(1);

      destroyBufferPool();
    });

    it('releases buffer even on error', async () => {
      const { withBuffer, getBufferPool, destroyBufferPool } = await import('../../src/gpu/buffer-pool.js');

      await expect(withBuffer(1024, GPUBufferUsage.STORAGE, async () => {
        throw new Error('Test error');
      })).rejects.toThrow('Test error');

      const pool = getBufferPool();
      expect(pool.getStats().activeBuffers).toBe(0);

      destroyBufferPool();
    });
  });
});

describe('gpu/profiler', () => {
  let mockDevice;

  beforeEach(async () => {
    vi.resetModules();
    mockDevice = createMockDevice();
    mockDevice.features = new Set(['timestamp-query']);
    const { getDevice, hasFeature, getDeviceLimits } = await import('../../src/gpu/device.js');
    getDevice.mockReturnValue(mockDevice);
    getDeviceLimits.mockReturnValue({
      maxBufferSize: 2147483647,
      maxStorageBufferBindingSize: 2147483647,
    });
    hasFeature.mockImplementation((f) => mockDevice.features.has(f));
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('GPUProfiler class', () => {
    it('tracks CPU timing with begin/end', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      profiler.begin('test_op');
      await new Promise((resolve) => setTimeout(resolve, 10));
      profiler.end('test_op');

      const result = profiler.getResult('test_op');
      expect(result).not.toBeNull();
      expect(result.count).toBe(1);
      expect(result.avg).toBeGreaterThan(0);
    });

    it('accumulates multiple measurements', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      for (let i = 0; i < 3; i++) {
        profiler.begin('repeated');
        profiler.end('repeated');
      }

      const result = profiler.getResult('repeated');
      expect(result.count).toBe(3);
    });

    it('tracks min and max times', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      profiler.begin('minmax');
      profiler.end('minmax');

      profiler.begin('minmax');
      await new Promise((resolve) => setTimeout(resolve, 5));
      profiler.end('minmax');

      const result = profiler.getResult('minmax');
      expect(result.max).toBeGreaterThanOrEqual(result.min);
    });

    it('warns on duplicate begin', async () => {
      const { log } = await import('../../src/debug/index.js');
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      profiler.begin('dup');
      profiler.begin('dup');

      expect(log.warn).toHaveBeenCalledWith('GPUProfiler', expect.stringContaining('already active'));
    });

    it('warns on end without begin', async () => {
      const { log } = await import('../../src/debug/index.js');
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      profiler.end('nonexistent');

      expect(log.warn).toHaveBeenCalledWith('GPUProfiler', expect.stringContaining('No active measurement'));
    });

    it('getResults returns all labeled results', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      profiler.begin('op1');
      profiler.end('op1');
      profiler.begin('op2');
      profiler.end('op2');

      const results = profiler.getResults();
      expect(Object.keys(results)).toContain('op1');
      expect(Object.keys(results)).toContain('op2');
    });

    it('getResult returns null for unknown label', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      expect(profiler.getResult('unknown')).toBeNull();
    });

    it('reset clears all data', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      profiler.begin('test');
      profiler.end('test');
      profiler.reset();

      expect(profiler.getResult('test')).toBeNull();
    });

    it('getReport formats results as string', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      profiler.begin('test');
      profiler.end('test');

      const report = profiler.getReport();
      expect(report).toContain('test');
      expect(report).toContain('Avg');
    });

    it('getReport handles empty results', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      const report = profiler.getReport();
      expect(report).toContain('No profiling data');
    });

    it('destroy cleans up resources', async () => {
      const { GPUProfiler } = await import('../../src/gpu/profiler.js');
      const profiler = new GPUProfiler(mockDevice);

      profiler.begin('test');
      profiler.end('test');
      profiler.destroy();

      const results = profiler.getResults();
      expect(Object.keys(results).length).toBe(0);
    });

    it('isGPUTimingAvailable reflects timestamp query support', async () => {
      const { GPUProfiler, createProfiler } = await import('../../src/gpu/profiler.js');

      const deviceWithTimestamp = {
        ...createMockDevice(),
        features: new Set(['timestamp-query']),
        createQuerySet: vi.fn(() => ({ destroy: vi.fn() })),
        createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
      };
      const profilerWith = new GPUProfiler(deviceWithTimestamp);
      expect(profilerWith.isGPUTimingAvailable()).toBe(true);

      const deviceWithoutTimestamp = {
        ...createMockDevice(),
        features: new Set([]),
      };
      const profilerWithout = new GPUProfiler(deviceWithoutTimestamp);
      expect(profilerWithout.isGPUTimingAvailable()).toBe(false);
    });
  });

  describe('global profiler', () => {
    it('getProfiler returns singleton', async () => {
      const { getProfiler } = await import('../../src/gpu/profiler.js');

      const profiler1 = getProfiler();
      const profiler2 = getProfiler();

      expect(profiler1).toBe(profiler2);
    });

    it('createProfiler creates new instance', async () => {
      const { createProfiler, getProfiler } = await import('../../src/gpu/profiler.js');

      const global = getProfiler();
      const custom = createProfiler(mockDevice);

      expect(custom).not.toBe(global);
    });
  });

  describe('timeOperation helper', () => {
    it('times async function and returns result', async () => {
      const { timeOperation } = await import('../../src/gpu/profiler.js');

      const { result, timeMs } = await timeOperation('async_test', async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return 42;
      });

      expect(result).toBe(42);
      expect(timeMs).toBeGreaterThan(0);
    });
  });
});
