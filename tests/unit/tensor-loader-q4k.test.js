import { describe, expect, it, vi } from 'vitest';

import { shouldUseFusedQ4K, loadQ4KDequant } from '../../src/loader/tensor-loader.js';
import { Q4K_BLOCK_BYTES, QK_K } from '../../src/loader/quantization-constants.js';
import { dequantize, dequantizeRowwise } from '../../src/gpu/kernel-selector.js';

vi.mock('../../src/debug/index.js', () => ({
  log: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
  trace: {
    loader: vi.fn(),
  },
}));

vi.mock('../../src/gpu/device.js', () => ({
  getDevice: vi.fn(() => ({
    queue: { writeBuffer: vi.fn() },
  })),
  getKernelCapabilities: vi.fn(() => ({
    hasF16: true,
    hasSubgroups: true,
  })),
}));

vi.mock('../../src/memory/buffer-pool.js', () => ({
  acquireBuffer: vi.fn((size) => ({ size })),
  releaseBuffer: vi.fn(),
}));

vi.mock('../../src/gpu/weight-buffer.js', () => ({
  createWeightBuffer: vi.fn((buffer, dtype, layout, shape, name) => ({
    buffer,
    dtype,
    layout,
    shape,
    name,
  })),
}));

vi.mock('../../src/gpu/kernel-selector.js', () => ({
  dequantize: vi.fn(async () => ({ buffer: { size: 1 } })),
  dequantizeRowwise: vi.fn(async () => ({ buffer: { size: 1 } })),
  dequantizeQ6K: vi.fn(),
  castF16ToF32: vi.fn(),
  runBF16ToF16: vi.fn(),
}));

function makeLocation(rows, K) {
  return {
    shape: [rows, K],
    size: rows * Math.ceil(K / QK_K) * Q4K_BLOCK_BYTES,
    role: 'matmul',
    dtype: 'Q4_K',
  };
}

function makeConfig(overrides = {}) {
  return {
    useFusedQ4K: true,
    keepF32Weights: false,
    allowF32UpcastNonMatmul: false,
    q4kLayout: 'row',
    gpuCapabilities: { hasF16: true, hasSubgroups: true },
    ...overrides,
  };
}

describe('tensor-loader Q4K alignment handling', () => {
  it('allows fused Q4K even when K is not 256-aligned', () => {
    const location = makeLocation(1, 1152);
    const config = makeConfig();
    expect(shouldUseFusedQ4K(location, config)).toBe(true);
  });

  it('allows fused Q4K when K is 256-aligned', () => {
    const location = makeLocation(1, 1024);
    const config = makeConfig();
    expect(shouldUseFusedQ4K(location, config)).toBe(true);
  });
});

describe('tensor-loader Q4K dequant path', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('uses row-wise dequant when K is not 256-aligned', async () => {
    const location = makeLocation(2, 1152);
    const config = makeConfig();
    const shardData = new Uint8Array(location.size);

    await loadQ4KDequant(shardData, location, 'test.q4k', config);

    expect(dequantizeRowwise).toHaveBeenCalledWith(
      expect.anything(),
      2,
      1152,
      { outputDtype: 'f16' }
    );
    expect(dequantize).not.toHaveBeenCalled();
  });

  it('uses standard dequant when K is 256-aligned', async () => {
    const location = makeLocation(2, 1024);
    const config = makeConfig();
    const shardData = new Uint8Array(location.size);

    await loadQ4KDequant(shardData, location, 'test.q4k', config);

    expect(dequantize).toHaveBeenCalled();
    expect(dequantizeRowwise).not.toHaveBeenCalled();
  });

  it('keeps row-wise dequant when output dtype is f32', async () => {
    const location = makeLocation(2, 1152);
    const config = makeConfig({
      keepF32Weights: true,
      gpuCapabilities: { hasF16: false, hasSubgroups: true },
    });
    const shardData = new Uint8Array(location.size);

    await loadQ4KDequant(shardData, location, 'test.q4k', config);

    expect(dequantizeRowwise).toHaveBeenCalledWith(
      expect.anything(),
      2,
      1152,
      { outputDtype: 'f32' }
    );
  });
});
