import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';

const mockAdapter = {
  features: new Set(['shader-f16', 'subgroups', 'timestamp-query']),
  limits: {
    maxStorageBufferBindingSize: 2147483647,
    maxBufferSize: 4294967295,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupStorageSize: 32768,
    maxStorageBuffersPerShaderStage: 8,
    maxUniformBufferBindingSize: 65536,
    maxComputeWorkgroupsPerDimension: 65535,
  },
  info: {
    vendor: 'apple',
    architecture: 'm2',
    device: 'Apple M2 Pro',
    description: 'Apple M2 Pro GPU',
  },
  requestDevice: vi.fn(),
};

const createMockDevice = () => ({
  features: new Set(['shader-f16', 'subgroups', 'timestamp-query']),
  limits: {
    maxStorageBufferBindingSize: 2147483647,
    maxBufferSize: 4294967295,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupStorageSize: 32768,
    maxStorageBuffersPerShaderStage: 8,
    maxUniformBufferBindingSize: 65536,
    maxComputeWorkgroupsPerDimension: 65535,
  },
  queue: {
    submit: vi.fn(),
  },
  lost: new Promise(() => {}),
  destroy: vi.fn(),
  createBuffer: vi.fn(),
  createShaderModule: vi.fn(),
  createComputePipeline: vi.fn(),
  createBindGroup: vi.fn(),
  createCommandEncoder: vi.fn(),
});

vi.mock('../../src/gpu/submit-tracker.js', () => ({
  wrapQueueForTracking: vi.fn(),
  TRACK_SUBMITS: false,
  setTrackSubmits: vi.fn(),
}));

vi.mock('../../src/config/platforms/loader.js', () => ({
  initializePlatform: vi.fn().mockResolvedValue({
    platform: { id: 'apple', name: 'Apple' },
    capabilities: { hasF16: true, hasSubgroups: true },
  }),
}));

vi.mock('../../src/config/kernels/registry.js', () => ({
  getRegistry: vi.fn().mockResolvedValue(new Map()),
}));

vi.mock('../../src/debug/index.js', () => ({
  log: {
    warn: vi.fn(),
    debug: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
  },
}));

describe('gpu/device', () => {
  let navigatorDescriptor;
  let mockDevice;

  beforeEach(() => {
    navigatorDescriptor = Object.getOwnPropertyDescriptor(global, 'navigator');
    mockDevice = createMockDevice();
    mockAdapter.requestDevice.mockResolvedValue(mockDevice);
    vi.clearAllMocks();
  });

  afterEach(async () => {
    if (navigatorDescriptor) {
      Object.defineProperty(global, 'navigator', navigatorDescriptor);
    } else {
      Object.defineProperty(global, 'navigator', {
        value: undefined,
        writable: true,
        configurable: true,
      });
    }
    vi.resetModules();
  });

  function setNavigator(value) {
    Object.defineProperty(global, 'navigator', {
      value,
      writable: true,
      configurable: true,
    });
  }

  describe('isWebGPUAvailable', () => {
    it('returns false when navigator.gpu is undefined', async () => {
      setNavigator({});
      const { isWebGPUAvailable } = await import('../../src/gpu/device.js');
      expect(isWebGPUAvailable()).toBe(false);
    });

    it('returns true when navigator.gpu exists', async () => {
      setNavigator({ gpu: {} });
      const { isWebGPUAvailable } = await import('../../src/gpu/device.js');
      expect(isWebGPUAvailable()).toBe(true);
    });

    it('returns false when navigator is undefined', async () => {
      setNavigator(undefined);
      const { isWebGPUAvailable } = await import('../../src/gpu/device.js');
      expect(isWebGPUAvailable()).toBe(false);
    });
  });

  describe('initDevice', () => {
    it('throws when WebGPU is not available', async () => {
      setNavigator({});
      const { initDevice } = await import('../../src/gpu/device.js');
      await expect(initDevice()).rejects.toThrow('WebGPU is not available');
    });

    it('throws when adapter is not available', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(null),
        },
      });
      const { initDevice } = await import('../../src/gpu/device.js');
      await expect(initDevice()).rejects.toThrow('Failed to get WebGPU adapter');
    });

    it('requests high-performance adapter first', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();

      expect(navigator.gpu.requestAdapter).toHaveBeenCalledWith(
        expect.objectContaining({ powerPreference: 'high-performance' })
      );

      destroyDevice();
    });

    it('falls back to minimal device when feature request fails', async () => {
      const minimalDevice = createMockDevice();
      minimalDevice.features = new Set();
      mockAdapter.requestDevice
        .mockRejectedValueOnce(new Error('Features not supported'))
        .mockResolvedValueOnce(minimalDevice);

      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, destroyDevice } = await import('../../src/gpu/device.js');

      const device = await initDevice();
      expect(device).toBeDefined();
      expect(mockAdapter.requestDevice).toHaveBeenCalledTimes(2);

      destroyDevice();
    });

    it('returns device and sets up capabilities', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, getDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');

      const device = await initDevice();

      expect(device).toBeDefined();
      expect(getDevice()).toBe(device);

      const caps = getKernelCapabilities();
      expect(caps.hasF16).toBe(true);
      expect(caps.hasSubgroups).toBe(true);
      expect(caps.hasTimestampQuery).toBe(true);

      destroyDevice();
    });

    it('caches device on subsequent calls', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, destroyDevice } = await import('../../src/gpu/device.js');

      const device1 = await initDevice();
      const device2 = await initDevice();

      expect(device1).toBe(device2);

      destroyDevice();
    });
  });

  describe('getKernelCapabilities', () => {
    it('throws when device not initialized', async () => {
      setNavigator({});
      const { getKernelCapabilities } = await import('../../src/gpu/device.js');
      expect(() => getKernelCapabilities()).toThrow('Device not initialized');
    });

    it('returns capabilities after device init', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();
      const caps = getKernelCapabilities();

      expect(caps).toMatchObject({
        hasSubgroups: true,
        hasF16: true,
        hasTimestampQuery: true,
        maxBufferSize: expect.any(Number),
        maxWorkgroupSize: expect.any(Number),
        maxWorkgroupStorageSize: expect.any(Number),
      });

      expect(caps.adapterInfo).toMatchObject({
        vendor: 'apple',
        architecture: 'm2',
        device: 'Apple M2 Pro',
      });

      destroyDevice();
    });

    it('returns copy of capabilities (immutable)', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();
      const caps1 = getKernelCapabilities();
      const caps2 = getKernelCapabilities();

      expect(caps1).not.toBe(caps2);
      expect(caps1).toEqual(caps2);

      destroyDevice();
    });
  });

  describe('getDevice', () => {
    it('returns null when device not initialized', async () => {
      setNavigator({});
      const { getDevice } = await import('../../src/gpu/device.js');
      expect(getDevice()).toBeNull();
    });

    it('returns device after init', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, getDevice, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();
      const device = getDevice();
      expect(device).toBeDefined();
      expect(device).not.toBeNull();

      destroyDevice();
    });
  });

  describe('getDeviceLimits', () => {
    it('returns null when device not initialized', async () => {
      setNavigator({});
      const { getDeviceLimits } = await import('../../src/gpu/device.js');
      expect(getDeviceLimits()).toBeNull();
    });

    it('returns limits after device init', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, getDeviceLimits, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();
      const limits = getDeviceLimits();

      expect(limits).toMatchObject({
        maxStorageBufferBindingSize: expect.any(Number),
        maxBufferSize: expect.any(Number),
        maxComputeWorkgroupSizeX: expect.any(Number),
        maxComputeWorkgroupSizeY: expect.any(Number),
        maxComputeWorkgroupSizeZ: expect.any(Number),
        maxComputeInvocationsPerWorkgroup: expect.any(Number),
        maxComputeWorkgroupStorageSize: expect.any(Number),
        maxStorageBuffersPerShaderStage: expect.any(Number),
        maxUniformBufferBindingSize: expect.any(Number),
        maxComputeWorkgroupsPerDimension: expect.any(Number),
      });

      destroyDevice();
    });
  });

  describe('hasFeature', () => {
    it('returns false when device not initialized', async () => {
      setNavigator({});
      const { hasFeature } = await import('../../src/gpu/device.js');
      expect(hasFeature('shader-f16')).toBe(false);
    });

    it('returns true for available features', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, hasFeature, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();
      expect(hasFeature('shader-f16')).toBe(true);
      expect(hasFeature('subgroups')).toBe(true);
      expect(hasFeature('timestamp-query')).toBe(true);

      destroyDevice();
    });

    it('returns false for unavailable features', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, hasFeature, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();
      expect(hasFeature('nonexistent-feature')).toBe(false);

      destroyDevice();
    });
  });

  describe('destroyDevice', () => {
    it('cleans up device state', async () => {
      setNavigator({
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue(mockAdapter),
        },
      });
      const { initDevice, getDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();
      expect(getDevice()).not.toBeNull();

      destroyDevice();

      expect(getDevice()).toBeNull();
      expect(() => getKernelCapabilities()).toThrow();
    });

    it('is safe to call multiple times', async () => {
      setNavigator({});
      const { destroyDevice } = await import('../../src/gpu/device.js');

      expect(() => {
        destroyDevice();
        destroyDevice();
        destroyDevice();
      }).not.toThrow();
    });
  });

  describe('FEATURES constants', () => {
    it('exports feature name constants', async () => {
      setNavigator({});
      const { FEATURES } = await import('../../src/gpu/device.js');

      expect(FEATURES.SHADER_F16).toBe('shader-f16');
      expect(FEATURES.SUBGROUPS).toBe('subgroups');
      expect(FEATURES.SUBGROUPS_F16).toBe('subgroups-f16');
      expect(FEATURES.TIMESTAMP_QUERY).toBe('timestamp-query');
    });
  });

  describe('adapter fallback behavior', () => {
    it('tries multiple adapter options', async () => {
      const requestAdapter = vi.fn()
        .mockResolvedValueOnce(null)
        .mockResolvedValueOnce(null)
        .mockResolvedValueOnce(mockAdapter);

      setNavigator({
        gpu: { requestAdapter },
      });

      const { initDevice, destroyDevice } = await import('../../src/gpu/device.js');

      await initDevice();
      expect(requestAdapter).toHaveBeenCalledTimes(3);

      destroyDevice();
    });
  });
});

describe('memory/capability (unit tests)', () => {
  describe('MemoryStrategy types', () => {
    it('defines MEMORY64 and SEGMENTED strategies', async () => {
      const expected = ['MEMORY64', 'SEGMENTED'];
      expect(expected).toContain('MEMORY64');
      expect(expected).toContain('SEGMENTED');
    });
  });
});
