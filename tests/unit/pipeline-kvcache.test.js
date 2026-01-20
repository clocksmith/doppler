import { describe, expect, it, beforeEach, afterEach, vi, beforeAll, afterAll } from 'vitest';
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  KVCache,
  SlidingWindowKVCache,
  isContiguousLayer,
  isPagedLayer,
  f32ToF16Bits,
  f16ToF32Bits,
  f32ToF16Array,
  f16ToF32Array,
} from '../../src/inference/kv-cache/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, '..', 'fixtures', 'mini-model');

const KB = 1024;
const MB = 1024 * 1024;
const DEFAULT_KV_CONFIG = {
  useGPU: false,
  layout: 'contiguous',
  pageSize: 256,
  kvDtype: 'f16',
  windowSize: 1024,
};

class MockGPUBuffer {
  constructor(size) {
    this.size = size;
    this.destroyed = false;
  }
  destroy() {
    this.destroyed = true;
  }
  getMappedRange() {
    return new ArrayBuffer(this.size);
  }
  unmap() {}
  async mapAsync() {}
}

globalThis.GPUBuffer = MockGPUBuffer;
globalThis.GPUBufferUsage = {
  STORAGE: 0x80,
  COPY_DST: 0x08,
  COPY_SRC: 0x04,
  MAP_READ: 0x01,
};
globalThis.GPUMapMode = {
  READ: 0x01,
};

vi.mock('../../src/gpu/device.js', () => ({
  getDevice: () => null,
}));

vi.mock('../../src/gpu/perf-guards.js', () => ({
  allowReadback: () => true,
}));

vi.mock('../../src/debug/index.js', () => ({
  log: {
    info: () => {},
    warn: () => {},
    error: () => {},
  },
  trace: {
    kernels: () => {},
    attn: () => {},
  },
}));

vi.mock('../../src/config/runtime.js', () => ({
  getRuntimeConfig: () => ({
    inference: {
      kvcache: {
        maxSeqLen: 4096,
        kvDtype: 'f16',
        layout: 'contiguous',
        pageSize: 256,
        windowSize: 1024,
      },
    },
  }),
}));

function createKVCacheConfig(overrides = {}) {
  return { ...DEFAULT_KV_CONFIG, ...overrides };
}

function createKVCache(overrides) {
  return new KVCache(createKVCacheConfig(overrides));
}

function createSlidingKVCache(overrides) {
  return new SlidingWindowKVCache(createKVCacheConfig(overrides));
}

describe('inference/kv-cache', () => {
  describe('KVCacheConfig parsing', () => {
    it('creates cache with required config fields', () => {
      const cache = createKVCache({
        numLayers: 4,
        numHeads: 8,
        headDim: 64,
        maxSeqLen: 2048,
      });

      expect(cache.numLayers).toBe(4);
      expect(cache.numHeads).toBe(8);
      expect(cache.headDim).toBe(64);
      expect(cache.maxSeqLen).toBe(2048);
    });

    it('uses default layout from runtime config when not specified', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
      });

      expect(cache.layout).toBe('contiguous');
    });

    it('uses specified layout over default', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(cache.layout).toBe('paged');
    });

    it('uses default kvDtype from runtime config when not specified', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
      });

      expect(cache.kvDtype).toBe('f16');
    });

    it('uses specified kvDtype over default', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        kvDtype: 'f32',
      });

      expect(cache.kvDtype).toBe('f32');
    });

    it('uses default pageSize from runtime config when not specified', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(cache.pageSize).toBe(256);
    });

    it('uses specified pageSize over default', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
        pageSize: 128,
      });

      expect(cache.pageSize).toBe(128);
    });
  });

  describe('cache size calculation', () => {
    it('calculates kvSize correctly', () => {
      const numHeads = 8;
      const headDim = 64;
      const cache = createKVCache({
        numLayers: 4,
        numHeads,
        headDim,
        maxSeqLen: 1024,
      });

      expect(cache.kvSize).toBe(numHeads * headDim);
    });

    it('calculates bytesPerElem correctly for f16', () => {
      const cache = createKVCache({
        numLayers: 4,
        numHeads: 8,
        headDim: 64,
        maxSeqLen: 1024,
        kvDtype: 'f16',
      });

      expect(cache.bytesPerElem).toBe(2);
    });

    it('calculates bytesPerElem correctly for f32', () => {
      const cache = createKVCache({
        numLayers: 4,
        numHeads: 8,
        headDim: 64,
        maxSeqLen: 1024,
        kvDtype: 'f32',
      });

      expect(cache.bytesPerElem).toBe(4);
    });

    it('calculates theoretical memory in getMemoryStats', () => {
      const numLayers = 4;
      const numHeads = 8;
      const headDim = 64;
      const maxSeqLen = 1024;
      const bytesPerElem = 4;

      const cache = createKVCache({
        numLayers,
        numHeads,
        headDim,
        maxSeqLen,
        kvDtype: 'f32',
      });

      const stats = cache.getMemoryStats();
      const expected = numLayers * 2 * maxSeqLen * (numHeads * headDim) * bytesPerElem;

      expect(stats.theoretical).toBe(expected);
    });

    it('tracks memory usage correctly for contiguous layout', () => {
      const numLayers = 2;
      const numHeads = 4;
      const headDim = 32;
      const maxSeqLen = 512;
      const bytesPerElem = 4;

      const cache = createKVCache({
        numLayers,
        numHeads,
        headDim,
        maxSeqLen,
        kvDtype: 'f32',
        layout: 'contiguous',
      });

      const sizePerLayer = maxSeqLen * (numHeads * headDim);
      const bytesPerLayer = sizePerLayer * bytesPerElem * 2;
      const expectedTotal = numLayers * bytesPerLayer;

      expect(cache.memoryUsage).toBe(expectedTotal);
    });

    it('tracks memory usage as zero initially for paged layout', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(cache.memoryUsage).toBe(0);
    });
  });

  describe('position tracking', () => {
    it('starts with currentSeqLen of 0', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
      });

      expect(cache.currentSeqLen).toBe(0);
    });

    it('increments currentSeqLen after update on last layer', () => {
      const numHeads = 4;
      const headDim = 32;
      const numLayers = 2;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys = new Float32Array(kvSize * 3);
      const values = new Float32Array(kvSize * 3);

      cache.update(0, keys, values, 0);
      expect(cache.currentSeqLen).toBe(0);

      cache.update(1, keys, values, 0);
      expect(cache.currentSeqLen).toBe(3);
    });

    it('increments correctly on sequential updates', () => {
      const numHeads = 4;
      const headDim = 32;
      const numLayers = 2;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys5 = new Float32Array(kvSize * 5);
      const values5 = new Float32Array(kvSize * 5);
      cache.update(0, keys5, values5, 0);
      cache.update(1, keys5, values5, 0);
      expect(cache.currentSeqLen).toBe(5);

      const keys1 = new Float32Array(kvSize * 1);
      const values1 = new Float32Array(kvSize * 1);
      cache.update(0, keys1, values1, 5);
      cache.update(1, keys1, values1, 5);
      expect(cache.currentSeqLen).toBe(6);

      cache.update(0, keys1, values1, 6);
      cache.update(1, keys1, values1, 6);
      expect(cache.currentSeqLen).toBe(7);
    });

    it('updates layer seqLen independently', () => {
      const numHeads = 4;
      const headDim = 32;
      const numLayers = 3;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);

      cache.update(0, keys, values, 0);
      expect(cache.layers[0].seqLen).toBe(5);
      expect(cache.layers[1].seqLen).toBe(0);
      expect(cache.layers[2].seqLen).toBe(0);

      cache.update(1, keys, values, 0);
      expect(cache.layers[1].seqLen).toBe(5);
      expect(cache.layers[2].seqLen).toBe(0);
    });

    it('throws on cache overflow', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const maxSeqLen = 10;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen,
      });

      const keys = new Float32Array(kvSize * 15);
      const values = new Float32Array(kvSize * 15);

      expect(() => cache.update(0, keys, values, 0)).toThrow('Cache overflow');
    });

    it('throws on overflow from startPos', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const maxSeqLen = 10;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen,
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);

      expect(() => cache.update(0, keys, values, 8)).toThrow('Cache overflow');
    });
  });

  describe('cache dtype selection', () => {
    it('defaults to f16 from runtime config', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
      });

      expect(cache.kvDtype).toBe('f16');
      expect(cache.bytesPerElem).toBe(2);
    });

    it('respects explicit f32 dtype', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        kvDtype: 'f32',
      });

      expect(cache.kvDtype).toBe('f32');
      expect(cache.bytesPerElem).toBe(4);
    });

    it('respects explicit f16 dtype', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        kvDtype: 'f16',
      });

      expect(cache.kvDtype).toBe('f16');
      expect(cache.bytesPerElem).toBe(2);
    });
  });

  describe('cache layout', () => {
    it('defaults to contiguous layout from runtime config', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
      });

      expect(cache.layout).toBe('contiguous');
    });

    it('creates contiguous layer structures', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      expect(isContiguousLayer(cache.layers[0])).toBe(true);
      expect(isPagedLayer(cache.layers[0])).toBe(false);
      expect(cache.layers[0].keys).toBeInstanceOf(Float32Array);
      expect(cache.layers[0].values).toBeInstanceOf(Float32Array);
    });

    it('creates paged layer structures', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(isPagedLayer(cache.layers[0])).toBe(true);
      expect(isContiguousLayer(cache.layers[0])).toBe(false);
      expect(cache.layers[0].keyPages).toBeDefined();
      expect(cache.layers[0].valuePages).toBeDefined();
    });

    it('reports layout in memory stats', () => {
      const contiguousCache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      const pagedCache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(contiguousCache.getMemoryStats().layout).toBe('contiguous');
      expect(pagedCache.getMemoryStats().layout).toBe('paged');
    });

    it('contiguous layer has correct array size', () => {
      const numHeads = 4;
      const headDim = 32;
      const maxSeqLen = 512;
      const expectedSize = maxSeqLen * numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen,
        layout: 'contiguous',
      });

      expect(cache.layers[0].keys.length).toBe(expectedSize);
      expect(cache.layers[0].values.length).toBe(expectedSize);
    });

    it('paged layout initializes with null pages', () => {
      const pageSize = 64;
      const maxSeqLen = 256;
      const expectedPages = Math.ceil(maxSeqLen / pageSize);

      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen,
        layout: 'paged',
        pageSize,
      });

      const layer = cache.layers[0];
      expect(layer.keyPages.length).toBe(expectedPages);
      expect(layer.valuePages.length).toBe(expectedPages);
      expect(layer.keyPages[0]).toBeNull();
      expect(layer.valuePages[0]).toBeNull();
    });
  });

  describe('cache reset/clear', () => {
    it('resets currentSeqLen to 0', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);
      expect(cache.currentSeqLen).toBe(5);

      cache.clear();
      expect(cache.currentSeqLen).toBe(0);
    });

    it('resets layer seqLen to 0', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      cache.clear();

      expect(cache.layers[0].seqLen).toBe(0);
      expect(cache.layers[1].seqLen).toBe(0);
    });

    it('zeros out contiguous arrays on clear', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);
      keys.fill(1.0);
      values.fill(2.0);
      cache.update(0, keys, values, 0);

      cache.clear();

      const layer = cache.layers[0];
      expect(layer.keys[0]).toBe(0);
      expect(layer.values[0]).toBe(0);
    });

    it('allows updates after clear', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      cache.clear();

      const keys2 = new Float32Array(kvSize * 3);
      const values2 = new Float32Array(kvSize * 3);
      cache.update(0, keys2, values2, 0);
      cache.update(1, keys2, values2, 0);

      expect(cache.currentSeqLen).toBe(3);
    });

    it('truncate reduces seqLen', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys = new Float32Array(kvSize * 10);
      const values = new Float32Array(kvSize * 10);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);
      expect(cache.currentSeqLen).toBe(10);

      cache.truncate(5);
      expect(cache.currentSeqLen).toBe(5);
      expect(cache.layers[0].seqLen).toBe(5);
      expect(cache.layers[1].seqLen).toBe(5);
    });

    it('truncate does nothing if length >= currentSeqLen', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      cache.truncate(10);
      expect(cache.currentSeqLen).toBe(5);

      cache.truncate(5);
      expect(cache.currentSeqLen).toBe(5);
    });
  });

  describe('prefill vs decode mode handling', () => {
    it('handles prefill with multiple tokens', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const prefillTokens = 64;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const keys = new Float32Array(kvSize * prefillTokens);
      const values = new Float32Array(kvSize * prefillTokens);
      keys.fill(0.5);
      values.fill(0.75);

      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      expect(cache.currentSeqLen).toBe(prefillTokens);
      expect(cache.layers[0].seqLen).toBe(prefillTokens);
    });

    it('handles decode with single token', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const prefillTokens = 64;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      const prefillKeys = new Float32Array(kvSize * prefillTokens);
      const prefillValues = new Float32Array(kvSize * prefillTokens);
      cache.update(0, prefillKeys, prefillValues, 0);
      cache.update(1, prefillKeys, prefillValues, 0);

      const decodeKeys = new Float32Array(kvSize * 1);
      const decodeValues = new Float32Array(kvSize * 1);
      decodeKeys.fill(1.0);
      decodeValues.fill(1.0);

      cache.update(0, decodeKeys, decodeValues, prefillTokens);
      cache.update(1, decodeKeys, decodeValues, prefillTokens);

      expect(cache.currentSeqLen).toBe(prefillTokens + 1);
    });

    it('get retrieves correct range after prefill', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      const keys = new Float32Array(kvSize * 10);
      const values = new Float32Array(kvSize * 10);
      for (let i = 0; i < kvSize * 10; i++) {
        keys[i] = i;
        values[i] = i + 1000;
      }

      cache.update(0, keys, values, 0);

      const result = cache.get(0, 0, 10);
      expect(result.keys.length).toBe(kvSize * 10);
      expect(result.values.length).toBe(kvSize * 10);
      expect(result.keys[0]).toBe(0);
      expect(result.values[0]).toBe(1000);
    });

    it('get retrieves correct range after decode', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      const prefillKeys = new Float32Array(kvSize * 5);
      const prefillValues = new Float32Array(kvSize * 5);
      prefillKeys.fill(1);
      prefillValues.fill(2);
      cache.update(0, prefillKeys, prefillValues, 0);

      const decodeKeys = new Float32Array(kvSize * 1);
      const decodeValues = new Float32Array(kvSize * 1);
      decodeKeys.fill(99);
      decodeValues.fill(100);
      cache.update(0, decodeKeys, decodeValues, 5);

      const result = cache.get(0, 5, 6);
      expect(result.keys[0]).toBe(99);
      expect(result.values[0]).toBe(100);
    });
  });

  describe('type guards', () => {
    it('isContiguousLayer returns true for contiguous layers', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      expect(isContiguousLayer(cache.layers[0])).toBe(true);
    });

    it('isContiguousLayer returns false for paged layers', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(isContiguousLayer(cache.layers[0])).toBe(false);
    });

    it('isPagedLayer returns true for paged layers', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(isPagedLayer(cache.layers[0])).toBe(true);
    });

    it('isPagedLayer returns false for contiguous layers', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      expect(isPagedLayer(cache.layers[0])).toBe(false);
    });
  });

  describe('F16 conversion utilities', () => {
    it('f32ToF16Bits converts zero correctly', () => {
      const result = f32ToF16Bits(0);
      expect(result).toBe(0);
    });

    it('f32ToF16Bits converts one correctly', () => {
      const result = f32ToF16Bits(1.0);
      const converted = f16ToF32Bits(result);
      expect(converted).toBeCloseTo(1.0, 2);
    });

    it('f32ToF16Bits handles negative values', () => {
      const result = f32ToF16Bits(-1.0);
      const converted = f16ToF32Bits(result);
      expect(converted).toBeCloseTo(-1.0, 2);
    });

    it('f16ToF32Bits converts zero correctly', () => {
      const result = f16ToF32Bits(0);
      expect(result).toBe(0);
    });

    it('f32ToF16Array converts array correctly', () => {
      const input = new Float32Array([0, 1.0, -1.0, 0.5]);
      const f16 = f32ToF16Array(input);
      const result = f16ToF32Array(f16);

      expect(result.length).toBe(4);
      expect(result[0]).toBeCloseTo(0, 2);
      expect(result[1]).toBeCloseTo(1.0, 2);
      expect(result[2]).toBeCloseTo(-1.0, 2);
      expect(result[3]).toBeCloseTo(0.5, 2);
    });

    it('f16ToF32Array converts array correctly', () => {
      const input = new Float32Array([1.5, 2.25, -3.5]);
      const f16 = f32ToF16Array(input);
      const result = f16ToF32Array(f16);

      expect(result.length).toBe(3);
      expect(result[0]).toBeCloseTo(1.5, 2);
      expect(result[1]).toBeCloseTo(2.25, 2);
      expect(result[2]).toBeCloseTo(-3.5, 2);
    });
  });

  describe('clone functionality', () => {
    it('creates independent copy of cache', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);
      keys.fill(42);
      values.fill(43);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      const cloned = cache.clone();

      expect(cloned.currentSeqLen).toBe(cache.currentSeqLen);
      expect(cloned.numLayers).toBe(cache.numLayers);
      expect(cloned.numHeads).toBe(cache.numHeads);
      expect(cloned.headDim).toBe(cache.headDim);
    });

    it('cloned cache is independent', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        layout: 'contiguous',
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);
      keys.fill(42);
      values.fill(43);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      const cloned = cache.clone();

      cache.clear();

      expect(cache.currentSeqLen).toBe(0);
      expect(cloned.currentSeqLen).toBe(5);
    });
  });

  describe('memory stats', () => {
    it('returns correct shape', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
      });

      const stats = cache.getMemoryStats();

      expect(stats).toHaveProperty('theoretical');
      expect(stats).toHaveProperty('allocated');
      expect(stats).toHaveProperty('used');
      expect(stats).toHaveProperty('efficiency');
      expect(stats).toHaveProperty('seqLen');
      expect(stats).toHaveProperty('maxSeqLen');
      expect(stats).toHaveProperty('layout');
    });

    it('seqLen matches currentSeqLen', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
      });

      expect(cache.getMemoryStats().seqLen).toBe(0);

      const keys = new Float32Array(kvSize * 10);
      const values = new Float32Array(kvSize * 10);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      expect(cache.getMemoryStats().seqLen).toBe(10);
    });

    it('maxSeqLen matches config', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 1024,
      });

      expect(cache.getMemoryStats().maxSeqLen).toBe(1024);
    });
  });
});

describe('inference/kv-cache/sliding-window', () => {
  describe('SlidingWindowKVCache construction', () => {
    it('creates cache with windowSize', () => {
      const cache = createSlidingKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        windowSize: 256,
      });

      expect(cache.windowSize).toBe(256);
    });

    it('uses default windowSize from runtime config when not specified', () => {
      const cache = createSlidingKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
      });

      expect(cache.windowSize).toBe(1024);
    });

    it('initializes totalTokensSeen to 0', () => {
      const cache = createSlidingKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        windowSize: 256,
      });

      expect(cache.totalTokensSeen).toBe(0);
    });
  });

  describe('SlidingWindowKVCache memory stats', () => {
    it('includes windowSize in memory stats', () => {
      const cache = createSlidingKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        windowSize: 256,
      });

      const stats = cache.getMemoryStats();
      expect(stats.windowSize).toBe(256);
    });

    it('includes totalTokensSeen in memory stats', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;

      const cache = createSlidingKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        windowSize: 256,
      });

      const keys = new Float32Array(kvSize * 10);
      const values = new Float32Array(kvSize * 10);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      const stats = cache.getMemoryStats();
      expect(stats.totalTokensSeen).toBe(20);
    });
  });

  describe('SlidingWindowKVCache sliding behavior', () => {
    it('does not slide when under window size', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const windowSize = 100;

      const cache = createSlidingKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        windowSize,
      });

      const keys = new Float32Array(kvSize * 50);
      const values = new Float32Array(kvSize * 50);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      expect(cache.currentSeqLen).toBe(50);
    });

    it('slides window when exceeding window size', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const windowSize = 50;

      const cache = createSlidingKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        windowSize,
      });

      const keys = new Float32Array(kvSize * 30);
      const values = new Float32Array(kvSize * 30);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      expect(cache.currentSeqLen).toBe(30);

      const keys2 = new Float32Array(kvSize * 30);
      const values2 = new Float32Array(kvSize * 30);
      cache.update(0, keys2, values2, cache.currentSeqLen);
      cache.update(1, keys2, values2, cache.currentSeqLen);

      expect(cache.currentSeqLen).toBeLessThanOrEqual(windowSize);
    });

    it('tracks totalTokensSeen correctly', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const windowSize = 50;

      const cache = createSlidingKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 512,
        windowSize,
      });

      const keys = new Float32Array(kvSize * 30);
      const values = new Float32Array(kvSize * 30);
      cache.update(0, keys, values, 0);
      cache.update(1, keys, values, 0);

      expect(cache.totalTokensSeen).toBe(60);

      const keys2 = new Float32Array(kvSize * 20);
      const values2 = new Float32Array(kvSize * 20);
      cache.update(0, keys2, values2, cache.currentSeqLen);
      cache.update(1, keys2, values2, cache.currentSeqLen);

      expect(cache.totalTokensSeen).toBe(100);
    });
  });
});

describe('paged layout operations', () => {
  describe('page allocation', () => {
    it('allocates pages lazily on update', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const pageSize = 16;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 128,
        layout: 'paged',
        pageSize,
      });

      expect(cache.layers[0].keyPages[0]).toBeNull();
      expect(cache.layers[0].allocatedPages).toBe(0);

      const keys = new Float32Array(kvSize * 8);
      const values = new Float32Array(kvSize * 8);
      cache.update(0, keys, values, 0);

      expect(cache.layers[0].keyPages[0]).not.toBeNull();
      expect(cache.layers[0].allocatedPages).toBe(1);
    });

    it('allocates multiple pages when needed', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const pageSize = 16;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 128,
        layout: 'paged',
        pageSize,
      });

      const keys = new Float32Array(kvSize * 32);
      const values = new Float32Array(kvSize * 32);
      cache.update(0, keys, values, 0);

      expect(cache.layers[0].allocatedPages).toBe(2);
    });

    it('tracks memory usage as pages are allocated', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const pageSize = 16;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 128,
        layout: 'paged',
        pageSize,
      });

      expect(cache.memoryUsage).toBe(0);

      const keys = new Float32Array(kvSize * 8);
      const values = new Float32Array(kvSize * 8);
      cache.update(0, keys, values, 0);

      const pageElements = pageSize * kvSize;
      const expectedUsage = pageElements * 4 * 2;
      expect(cache.memoryUsage).toBe(expectedUsage);
    });
  });

  describe('paged get operation', () => {
    it('retrieves data from paged layout correctly', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const pageSize = 16;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 128,
        layout: 'paged',
        pageSize,
      });

      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);
      for (let i = 0; i < keys.length; i++) {
        keys[i] = i;
        values[i] = i + 1000;
      }
      cache.update(0, keys, values, 0);

      const result = cache.get(0, 0, 5);
      expect(result.keys.length).toBe(kvSize * 5);
      expect(result.values.length).toBe(kvSize * 5);
      expect(result.keys[0]).toBe(0);
      expect(result.values[0]).toBe(1000);
    });

    it('retrieves data spanning multiple pages', () => {
      const numHeads = 4;
      const headDim = 32;
      const kvSize = numHeads * headDim;
      const pageSize = 8;

      const cache = createKVCache({
        numLayers: 2,
        numHeads,
        headDim,
        maxSeqLen: 128,
        layout: 'paged',
        pageSize,
      });

      const keys = new Float32Array(kvSize * 20);
      const values = new Float32Array(kvSize * 20);
      for (let i = 0; i < keys.length; i++) {
        keys[i] = i;
        values[i] = i + 1000;
      }
      cache.update(0, keys, values, 0);

      const result = cache.get(0, 0, 20);
      expect(result.keys.length).toBe(kvSize * 20);

      expect(result.keys[0]).toBe(0);
      expect(result.keys[kvSize * pageSize]).toBe(kvSize * pageSize);
    });
  });
});

describe('GPU cache helpers', () => {
  describe('hasGPUCache', () => {
    it('returns false when useGPU is false', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        useGPU: false,
      });

      expect(cache.hasGPUCache()).toBe(false);
    });
  });

  describe('getGPUBuffers', () => {
    it('returns null when no GPU buffers exist', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        useGPU: false,
      });

      expect(cache.getGPUBuffers(0)).toBeNull();
    });

    it('returns null for paged layout', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(cache.getGPUBuffers(0)).toBeNull();
    });
  });

  describe('getKeyCache and getValueCache', () => {
    it('returns CPU arrays for contiguous layout without GPU', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'contiguous',
        useGPU: false,
      });

      const keyCache = cache.getKeyCache(0);
      const valueCache = cache.getValueCache(0);

      expect(keyCache).toBeInstanceOf(Float32Array);
      expect(valueCache).toBeInstanceOf(Float32Array);
    });

    it('returns null for paged layout', () => {
      const cache = createKVCache({
        numLayers: 2,
        numHeads: 4,
        headDim: 32,
        maxSeqLen: 512,
        layout: 'paged',
      });

      expect(cache.getKeyCache(0)).toBeNull();
      expect(cache.getValueCache(0)).toBeNull();
    });
  });
});

describe('mini-model fixture integration', () => {
  let manifest;

  beforeAll(async () => {
    const manifestPath = join(FIXTURES_DIR, 'manifest.json');
    const content = await readFile(manifestPath, 'utf-8');
    manifest = JSON.parse(content);
  });

  describe('KV cache allocation from manifest', () => {
    it('creates cache matching mini-model architecture', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
      });

      expect(cache.numLayers).toBe(2);
      expect(cache.numHeads).toBe(2);
      expect(cache.headDim).toBe(32);
      expect(cache.maxSeqLen).toBe(128);
    });

    it('calculates correct kvSize from manifest dimensions', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
      });

      const expectedKvSize = arch.numKeyValueHeads * arch.headDim;
      expect(cache.kvSize).toBe(expectedKvSize);
      expect(cache.kvSize).toBe(64);
    });

    it('allocates correct memory for mini-model with f32', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        kvDtype: 'f32',
        layout: 'contiguous',
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const sizePerLayer = arch.maxSeqLen * kvSize;
      const bytesPerLayer = sizePerLayer * 4 * 2;
      const expectedMemory = arch.numLayers * bytesPerLayer;

      expect(cache.memoryUsage).toBe(expectedMemory);
    });

    it('allocates correct memory for mini-model with f16', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        kvDtype: 'f16',
        layout: 'contiguous',
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const sizePerLayer = arch.maxSeqLen * kvSize;
      const bytesPerLayer = sizePerLayer * 2 * 2;
      const expectedMemory = arch.numLayers * bytesPerLayer;

      expect(cache.memoryUsage).toBe(expectedMemory);
    });
  });

  describe('position tracking with mini-model', () => {
    it('handles prefill up to mini-model maxSeqLen', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const prefillLen = arch.maxSeqLen - 10;

      const keys = new Float32Array(kvSize * prefillLen);
      const values = new Float32Array(kvSize * prefillLen);

      for (let l = 0; l < arch.numLayers; l++) {
        cache.update(l, keys, values, 0);
      }

      expect(cache.currentSeqLen).toBe(prefillLen);
    });

    it('handles decode after prefill within mini-model limits', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const prefillLen = 50;

      const prefillKeys = new Float32Array(kvSize * prefillLen);
      const prefillValues = new Float32Array(kvSize * prefillLen);

      for (let l = 0; l < arch.numLayers; l++) {
        cache.update(l, prefillKeys, prefillValues, 0);
      }

      const decodeKeys = new Float32Array(kvSize);
      const decodeValues = new Float32Array(kvSize);

      for (let l = 0; l < arch.numLayers; l++) {
        cache.update(l, decodeKeys, decodeValues, prefillLen);
      }

      expect(cache.currentSeqLen).toBe(prefillLen + 1);
    });

    it('throws on overflow beyond mini-model maxSeqLen', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const overflowLen = arch.maxSeqLen + 10;

      const keys = new Float32Array(kvSize * overflowLen);
      const values = new Float32Array(kvSize * overflowLen);

      expect(() => cache.update(0, keys, values, 0)).toThrow('Cache overflow');
    });
  });

  describe('F16 vs F32 dtype with mini-model', () => {
    it('uses f16 bytesPerElem correctly', () => {
      const arch = manifest.architecture;
      const cacheF16 = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        kvDtype: 'f16',
      });

      expect(cacheF16.bytesPerElem).toBe(2);
      expect(cacheF16.kvDtype).toBe('f16');
    });

    it('uses f32 bytesPerElem correctly', () => {
      const arch = manifest.architecture;
      const cacheF32 = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        kvDtype: 'f32',
      });

      expect(cacheF32.bytesPerElem).toBe(4);
      expect(cacheF32.kvDtype).toBe('f32');
    });

    it('f16 cache uses half the memory of f32', () => {
      const arch = manifest.architecture;

      const cacheF16 = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        kvDtype: 'f16',
        layout: 'contiguous',
      });

      const cacheF32 = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        kvDtype: 'f32',
        layout: 'contiguous',
      });

      expect(cacheF16.memoryUsage).toBe(cacheF32.memoryUsage / 2);
    });
  });

  describe('cache update on decode with mini-model', () => {
    it('stores and retrieves values correctly', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        layout: 'contiguous',
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const keys = new Float32Array(kvSize * 5);
      const values = new Float32Array(kvSize * 5);

      for (let i = 0; i < keys.length; i++) {
        keys[i] = i * 0.1;
        values[i] = i * 0.2;
      }

      cache.update(0, keys, values, 0);

      const result = cache.get(0, 0, 5);

      expect(result.keys[0]).toBeCloseTo(0);
      expect(result.keys[1]).toBeCloseTo(0.1);
      expect(result.values[0]).toBeCloseTo(0);
      expect(result.values[1]).toBeCloseTo(0.2);
    });

    it('appends decode tokens correctly', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        layout: 'contiguous',
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;

      const prefillKeys = new Float32Array(kvSize * 10);
      const prefillValues = new Float32Array(kvSize * 10);
      prefillKeys.fill(1.0);
      prefillValues.fill(2.0);

      for (let l = 0; l < arch.numLayers; l++) {
        cache.update(l, prefillKeys, prefillValues, 0);
      }

      const decodeKeys = new Float32Array(kvSize);
      const decodeValues = new Float32Array(kvSize);
      decodeKeys.fill(99.0);
      decodeValues.fill(100.0);

      for (let l = 0; l < arch.numLayers; l++) {
        cache.update(l, decodeKeys, decodeValues, 10);
      }

      const result = cache.get(0, 10, 11);
      expect(result.keys[0]).toBeCloseTo(99.0);
      expect(result.values[0]).toBeCloseTo(100.0);
    });

    it('maintains layer independence during decode', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        layout: 'contiguous',
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;

      const keys0 = new Float32Array(kvSize * 5);
      const values0 = new Float32Array(kvSize * 5);
      keys0.fill(1.0);
      values0.fill(2.0);

      const keys1 = new Float32Array(kvSize * 5);
      const values1 = new Float32Array(kvSize * 5);
      keys1.fill(3.0);
      values1.fill(4.0);

      cache.update(0, keys0, values0, 0);
      cache.update(1, keys1, values1, 0);

      const result0 = cache.get(0, 0, 5);
      const result1 = cache.get(1, 0, 5);

      expect(result0.keys[0]).toBeCloseTo(1.0);
      expect(result0.values[0]).toBeCloseTo(2.0);
      expect(result1.keys[0]).toBeCloseTo(3.0);
      expect(result1.values[0]).toBeCloseTo(4.0);
    });
  });

  describe('sequence length limits with mini-model', () => {
    it('respects maxSeqLen from manifest', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
      });

      expect(cache.maxSeqLen).toBe(128);
      expect(cache.getMemoryStats().maxSeqLen).toBe(128);
    });

    it('allows exact maxSeqLen fill', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const keys = new Float32Array(kvSize * arch.maxSeqLen);
      const values = new Float32Array(kvSize * arch.maxSeqLen);

      expect(() => {
        for (let l = 0; l < arch.numLayers; l++) {
          cache.update(l, keys, values, 0);
        }
      }).not.toThrow();

      expect(cache.currentSeqLen).toBe(arch.maxSeqLen);
    });

    it('reports sequence length in memory stats', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const keys = new Float32Array(kvSize * 50);
      const values = new Float32Array(kvSize * 50);

      for (let l = 0; l < arch.numLayers; l++) {
        cache.update(l, keys, values, 0);
      }

      const stats = cache.getMemoryStats();
      expect(stats.seqLen).toBe(50);
      expect(stats.maxSeqLen).toBe(128);
    });

    it('calculates memory efficiency correctly', () => {
      const arch = manifest.architecture;
      const cache = createKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        kvDtype: 'f32',
        layout: 'contiguous',
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const fillLen = 64;
      const keys = new Float32Array(kvSize * fillLen);
      const values = new Float32Array(kvSize * fillLen);

      for (let l = 0; l < arch.numLayers; l++) {
        cache.update(l, keys, values, 0);
      }

      const stats = cache.getMemoryStats();
      const expectedEfficiency = fillLen / arch.maxSeqLen;

      expect(stats.efficiency).toBeCloseTo(expectedEfficiency, 2);
    });
  });

  describe('sliding window with mini-model', () => {
    it('creates sliding window cache from manifest', () => {
      const arch = manifest.architecture;
      const windowSize = 64;

      const cache = createSlidingKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        windowSize,
      });

      expect(cache.windowSize).toBe(windowSize);
      expect(cache.maxSeqLen).toBe(arch.maxSeqLen);
    });

    it('sliding window stats include totalTokensSeen', () => {
      const arch = manifest.architecture;
      const windowSize = 32;

      const cache = createSlidingKVCache({
        numLayers: arch.numLayers,
        numHeads: arch.numKeyValueHeads,
        headDim: arch.headDim,
        maxSeqLen: arch.maxSeqLen,
        windowSize,
      });

      const kvSize = arch.numKeyValueHeads * arch.headDim;
      const keys = new Float32Array(kvSize * 20);
      const values = new Float32Array(kvSize * 20);

      for (let l = 0; l < arch.numLayers; l++) {
        cache.update(l, keys, values, 0);
      }

      const stats = cache.getMemoryStats();
      expect(stats.totalTokensSeen).toBe(40);
      expect(stats.windowSize).toBe(windowSize);
    });
  });
});
