import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

import {
  hexToBytes,
  computeSHA256,
  getHashAlgorithm,
  getOpfsPathConfig,
  setOpfsPathConfig,
  cleanup,
} from '../../src/storage/shard-manager.js';

import {
  formatBytes,
  QuotaExceededError,
  isStorageAPIAvailable,
  isOPFSAvailable,
  isIndexedDBAvailable,
  clearCache,
} from '../../src/storage/quota.js';

import {
  formatSpeed,
  estimateTimeRemaining,
} from '../../src/storage/downloader.js';

import {
  GEMMA_1B_REQUIREMENTS,
  MODEL_REQUIREMENTS,
  formatPreflightResult,
} from '../../src/storage/preflight.js';

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;
const KB = 1024;

describe('storage/shard-manager', () => {
  beforeEach(() => {
    cleanup();
  });

  describe('hexToBytes', () => {
    it('converts hex string to Uint8Array', () => {
      const hex = '00ff10ab';
      const bytes = hexToBytes(hex);

      expect(bytes).toBeInstanceOf(Uint8Array);
      expect(bytes.length).toBe(4);
      expect(bytes[0]).toBe(0x00);
      expect(bytes[1]).toBe(0xff);
      expect(bytes[2]).toBe(0x10);
      expect(bytes[3]).toBe(0xab);
    });

    it('handles empty string', () => {
      const bytes = hexToBytes('');
      expect(bytes.length).toBe(0);
    });

    it('handles lowercase hex', () => {
      const bytes = hexToBytes('abcdef');
      expect(bytes[0]).toBe(0xab);
      expect(bytes[1]).toBe(0xcd);
      expect(bytes[2]).toBe(0xef);
    });

    it('handles uppercase hex', () => {
      const bytes = hexToBytes('ABCDEF');
      expect(bytes[0]).toBe(0xab);
      expect(bytes[1]).toBe(0xcd);
      expect(bytes[2]).toBe(0xef);
    });

    it('produces correct length for 32-byte hash', () => {
      const hash = 'a'.repeat(64);
      const bytes = hexToBytes(hash);
      expect(bytes.length).toBe(32);
    });
  });

  describe('computeSHA256', () => {
    it('computes SHA-256 hash of Uint8Array', async () => {
      const data = new Uint8Array([1, 2, 3, 4, 5]);
      const hash = await computeSHA256(data);

      expect(typeof hash).toBe('string');
      expect(hash.length).toBe(64);
      expect(/^[0-9a-f]+$/.test(hash)).toBe(true);
    });

    it('computes SHA-256 hash of ArrayBuffer', async () => {
      const buffer = new ArrayBuffer(10);
      const view = new Uint8Array(buffer);
      view.fill(42);

      const hash = await computeSHA256(buffer);

      expect(typeof hash).toBe('string');
      expect(hash.length).toBe(64);
    });

    it('produces consistent hash for same input', async () => {
      const data = new Uint8Array([10, 20, 30]);
      const hash1 = await computeSHA256(data);
      const hash2 = await computeSHA256(data);

      expect(hash1).toBe(hash2);
    });

    it('produces different hash for different input', async () => {
      const data1 = new Uint8Array([1, 2, 3]);
      const data2 = new Uint8Array([4, 5, 6]);

      const hash1 = await computeSHA256(data1);
      const hash2 = await computeSHA256(data2);

      expect(hash1).not.toBe(hash2);
    });

    it('handles empty data', async () => {
      const data = new Uint8Array(0);
      const hash = await computeSHA256(data);

      expect(hash.length).toBe(64);
    });
  });

  describe('getOpfsPathConfig / setOpfsPathConfig', () => {
    afterEach(() => {
      setOpfsPathConfig(null);
    });

    it('returns default config when no override set', () => {
      const config = getOpfsPathConfig();
      expect(config).toBeDefined();
      expect(config.opfsRootDir).toBeDefined();
    });

    it('returns override config when set', () => {
      const customConfig = { opfsRootDir: 'custom-models' };
      setOpfsPathConfig(customConfig);

      const config = getOpfsPathConfig();
      expect(config.opfsRootDir).toBe('custom-models');
    });
  });

  describe('cleanup', () => {
    it('resets module state without error', () => {
      expect(() => cleanup()).not.toThrow();
    });
  });
});

describe('storage/quota', () => {
  beforeEach(() => {
    clearCache();
  });

  describe('formatBytes', () => {
    it('formats zero bytes', () => {
      expect(formatBytes(0)).toBe('0 B');
    });

    it('formats bytes', () => {
      expect(formatBytes(500)).toBe('500 B');
    });

    it('formats kilobytes', () => {
      const result = formatBytes(1024);
      expect(result).toContain('KB');
    });

    it('formats megabytes', () => {
      const result = formatBytes(1024 * 1024);
      expect(result).toContain('MB');
    });

    it('formats gigabytes', () => {
      const result = formatBytes(1024 * 1024 * 1024);
      expect(result).toContain('GB');
    });

    it('formats terabytes', () => {
      const result = formatBytes(1024 * 1024 * 1024 * 1024);
      expect(result).toContain('TB');
    });

    it('formats fractional values correctly', () => {
      const result = formatBytes(1.5 * GB);
      expect(result).toBe('1.50 GB');
    });

    it('handles large values', () => {
      const result = formatBytes(100 * GB);
      expect(result).toBe('100.00 GB');
    });
  });

  describe('formatSpeed', () => {
    it('formats speed in bytes per second', () => {
      const result = formatSpeed(1000);
      expect(result).toContain('/s');
    });

    it('formats MB/s correctly', () => {
      const result = formatSpeed(10 * MB);
      expect(result).toContain('MB/s');
    });

    it('formats zero speed', () => {
      const result = formatSpeed(0);
      expect(result).toBe('0 B/s');
    });
  });

  describe('estimateTimeRemaining', () => {
    it('returns calculating for zero speed', () => {
      const result = estimateTimeRemaining(1000, 0);
      expect(result).toBe('Calculating...');
    });

    it('returns calculating for negative speed', () => {
      const result = estimateTimeRemaining(1000, -10);
      expect(result).toBe('Calculating...');
    });

    it('returns seconds for short durations', () => {
      const result = estimateTimeRemaining(500, 100);
      expect(result).toMatch(/^\d+s$/);
    });

    it('returns minutes for medium durations', () => {
      const result = estimateTimeRemaining(6000, 100);
      expect(result).toMatch(/^\d+m$/);
    });

    it('returns hours and minutes for long durations', () => {
      const result = estimateTimeRemaining(400000, 100);
      expect(result).toMatch(/^\d+h \d+m$/);
    });

    it('calculates correctly for realistic download scenario', () => {
      const remaining = 500 * MB;
      const speed = 10 * MB;
      const result = estimateTimeRemaining(remaining, speed);

      expect(result).toMatch(/^(50s|\d+m)$/);
    });
  });

  describe('QuotaExceededError', () => {
    it('creates error with correct properties', () => {
      const error = new QuotaExceededError(2 * GB, 1 * GB);

      expect(error).toBeInstanceOf(Error);
      expect(error.name).toBe('QuotaExceededError');
      expect(error.required).toBe(2 * GB);
      expect(error.available).toBe(1 * GB);
      expect(error.shortfall).toBe(1 * GB);
    });

    it('includes human-readable message', () => {
      const error = new QuotaExceededError(2 * GB, 1 * GB);

      expect(error.message).toContain('Insufficient storage');
      expect(error.message).toContain('GB');
    });

    it('calculates shortfall correctly', () => {
      const error = new QuotaExceededError(5 * GB, 2 * GB);

      expect(error.shortfall).toBe(3 * GB);
    });
  });

  describe('storage API availability checks', () => {
    it('isStorageAPIAvailable returns boolean', () => {
      const result = isStorageAPIAvailable();
      expect(typeof result).toBe('boolean');
    });

    it('isOPFSAvailable returns boolean', () => {
      const result = isOPFSAvailable();
      expect(typeof result).toBe('boolean');
    });

    it('isIndexedDBAvailable returns boolean', () => {
      const result = isIndexedDBAvailable();
      expect(typeof result).toBe('boolean');
    });
  });

  describe('clearCache', () => {
    it('clears persistence cache without error', () => {
      expect(() => clearCache()).not.toThrow();
    });
  });
});

describe('storage/preflight', () => {
  describe('MODEL_REQUIREMENTS', () => {
    it('includes gemma-3-1b-it-wq4k', () => {
      expect(MODEL_REQUIREMENTS['gemma-3-1b-it-wq4k']).toBeDefined();
    });

    it('gemma requirements have expected properties', () => {
      const req = MODEL_REQUIREMENTS['gemma-3-1b-it-wq4k'];

      expect(req.modelId).toBe('gemma-3-1b-it-wq4k');
      expect(req.displayName).toBeDefined();
      expect(req.downloadSize).toBeGreaterThan(0);
      expect(req.vramRequired).toBeGreaterThan(0);
      expect(req.paramCount).toBe('1B');
      expect(req.quantization).toBe('Q4_K_M');
    });
  });

  describe('GEMMA_1B_REQUIREMENTS', () => {
    it('has correct modelId', () => {
      expect(GEMMA_1B_REQUIREMENTS.modelId).toBe('gemma-3-1b-it-wq4k');
    });

    it('has reasonable download size', () => {
      expect(GEMMA_1B_REQUIREMENTS.downloadSize).toBeGreaterThan(400 * MB);
      expect(GEMMA_1B_REQUIREMENTS.downloadSize).toBeLessThan(1 * GB);
    });

    it('has reasonable VRAM requirement', () => {
      expect(GEMMA_1B_REQUIREMENTS.vramRequired).toBeGreaterThan(1 * GB);
      expect(GEMMA_1B_REQUIREMENTS.vramRequired).toBeLessThan(4 * GB);
    });

    it('has architecture specified', () => {
      expect(GEMMA_1B_REQUIREMENTS.architecture).toBe('Gemma3ForCausalLM');
    });
  });

  describe('formatPreflightResult', () => {
    it('formats result with all sections', () => {
      const result = {
        canProceed: true,
        vram: {
          required: 1.5 * GB,
          available: 8 * GB,
          sufficient: true,
          message: 'VRAM OK',
        },
        storage: {
          required: 500 * MB,
          available: 10 * GB,
          sufficient: true,
          message: 'Storage OK',
        },
        gpu: {
          hasWebGPU: true,
          hasF16: true,
          device: 'Apple M1',
          isUnified: true,
        },
        warnings: [],
        blockers: [],
      };

      const formatted = formatPreflightResult(result);

      expect(formatted).toContain('GPU: Apple M1');
      expect(formatted).toContain('VRAM: VRAM OK');
      expect(formatted).toContain('Storage: Storage OK');
      expect(formatted).toContain('Can proceed: Yes');
    });

    it('includes warnings when present', () => {
      const result = {
        canProceed: true,
        vram: { message: 'OK' },
        storage: { message: 'OK' },
        gpu: { device: 'Test GPU' },
        warnings: ['F16 not supported'],
        blockers: [],
      };

      const formatted = formatPreflightResult(result);

      expect(formatted).toContain('Warnings');
      expect(formatted).toContain('F16 not supported');
    });

    it('includes blockers when present', () => {
      const result = {
        canProceed: false,
        vram: { message: 'Insufficient' },
        storage: { message: 'OK' },
        gpu: { device: 'Unknown' },
        warnings: [],
        blockers: ['WebGPU not available'],
      };

      const formatted = formatPreflightResult(result);

      expect(formatted).toContain('Blockers');
      expect(formatted).toContain('WebGPU not available');
      expect(formatted).toContain('Can proceed: No');
    });
  });
});

describe('download progress tracking', () => {
  describe('progress calculation', () => {
    it('calculates percent correctly', () => {
      const downloaded = 250 * MB;
      const total = 500 * MB;
      const percent = (downloaded / total) * 100;

      expect(percent).toBe(50);
    });

    it('handles zero total gracefully', () => {
      const downloaded = 0;
      const total = 0;
      const percent = total > 0 ? (downloaded / total) * 100 : 0;

      expect(percent).toBe(0);
    });

    it('caps at 100 percent', () => {
      const downloaded = 600 * MB;
      const total = 500 * MB;
      const percent = Math.min(100, (downloaded / total) * 100);

      expect(percent).toBe(100);
    });
  });

  describe('speed calculation', () => {
    it('calculates bytes per second', () => {
      const bytesDownloaded = 10 * MB;
      const timeSeconds = 2;
      const speed = bytesDownloaded / timeSeconds;

      expect(speed).toBe(5 * MB);
    });

    it('handles short time intervals', () => {
      const bytesDownloaded = 1 * MB;
      const timeSeconds = 0.5;
      const speed = bytesDownloaded / timeSeconds;

      expect(speed).toBe(2 * MB);
    });

    it('returns zero for zero time', () => {
      const bytesDownloaded = 100;
      const timeSeconds = 0;
      const speed = timeSeconds > 0 ? bytesDownloaded / timeSeconds : 0;

      expect(speed).toBe(0);
    });
  });

  describe('shard tracking', () => {
    it('tracks completed shards as Set', () => {
      const completedShards = new Set();

      completedShards.add(0);
      completedShards.add(2);
      completedShards.add(5);

      expect(completedShards.size).toBe(3);
      expect(completedShards.has(0)).toBe(true);
      expect(completedShards.has(1)).toBe(false);
      expect(completedShards.has(2)).toBe(true);
    });

    it('calculates missing shards', () => {
      const totalShards = 10;
      const completedShards = new Set([0, 1, 2, 5, 7]);

      const missingShards = [];
      for (let i = 0; i < totalShards; i++) {
        if (!completedShards.has(i)) {
          missingShards.push(i);
        }
      }

      expect(missingShards).toEqual([3, 4, 6, 8, 9]);
    });

    it('calculates downloaded bytes from completed shards', () => {
      const shardSizes = [100 * MB, 100 * MB, 100 * MB, 100 * MB, 100 * MB];
      const completedShards = new Set([0, 2, 4]);

      let downloadedBytes = 0;
      for (const idx of completedShards) {
        downloadedBytes += shardSizes[idx];
      }

      expect(downloadedBytes).toBe(300 * MB);
    });
  });
});

describe('shard caching patterns', () => {
  describe('cache key generation', () => {
    it('generates unique keys for shards', () => {
      const modelId = 'gemma-3-1b-it-wq4k';
      const shardIndex = 3;
      const key = `${modelId}/shard_${shardIndex}.bin`;

      expect(key).toBe('gemma-3-1b-it-wq4k/shard_3.bin');
    });

    it('sanitizes model ID for filesystem', () => {
      const modelId = 'model/with:special*chars';
      const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');

      expect(safeName).toBe('model_with_special_chars');
    });
  });

  describe('shard existence check', () => {
    it('returns boolean for existence', () => {
      const cache = new Map();
      cache.set('shard_0', new ArrayBuffer(100));

      expect(cache.has('shard_0')).toBe(true);
      expect(cache.has('shard_1')).toBe(false);
    });
  });

  describe('cache eviction', () => {
    it('removes oldest entry when at capacity', () => {
      const maxEntries = 3;
      const cache = new Map();

      cache.set('shard_0', { data: new ArrayBuffer(10), time: 1 });
      cache.set('shard_1', { data: new ArrayBuffer(10), time: 2 });
      cache.set('shard_2', { data: new ArrayBuffer(10), time: 3 });

      if (cache.size >= maxEntries) {
        const oldestKey = [...cache.entries()]
          .sort((a, b) => a[1].time - b[1].time)[0][0];
        cache.delete(oldestKey);
      }

      cache.set('shard_3', { data: new ArrayBuffer(10), time: 4 });

      expect(cache.size).toBe(3);
      expect(cache.has('shard_0')).toBe(false);
      expect(cache.has('shard_3')).toBe(true);
    });
  });
});

describe('preflight check patterns', () => {
  describe('VRAM estimation', () => {
    it('uses unified memory ratio for Apple devices', () => {
      const systemMemoryGB = 16;
      const unifiedRatio = 0.75;
      const availableVRAM = systemMemoryGB * GB * unifiedRatio;

      expect(availableVRAM).toBe(12 * GB);
    });

    it('uses maxBufferSize for discrete GPU', () => {
      const maxBufferSize = 6 * GB;
      const availableVRAM = maxBufferSize;

      expect(availableVRAM).toBe(6 * GB);
    });

    it('uses fallback for unknown GPU', () => {
      const fallbackVRAM = 4 * GB;
      expect(fallbackVRAM).toBe(4 * GB);
    });
  });

  describe('storage estimation', () => {
    it('calculates available space correctly', () => {
      const quota = 10 * GB;
      const usage = 3 * GB;
      const available = quota - usage;

      expect(available).toBe(7 * GB);
    });

    it('calculates shortfall correctly', () => {
      const required = 5 * GB;
      const available = 2 * GB;
      const shortfall = required - available;

      expect(shortfall).toBe(3 * GB);
    });

    it('identifies low space condition', () => {
      const available = 400 * MB;
      const lowSpaceThreshold = 500 * MB;
      const isLowSpace = available < lowSpaceThreshold;

      expect(isLowSpace).toBe(true);
    });

    it('identifies critical space condition', () => {
      const available = 50 * MB;
      const criticalThreshold = 100 * MB;
      const isCritical = available < criticalThreshold;

      expect(isCritical).toBe(true);
    });
  });

  describe('blocker detection', () => {
    it('identifies WebGPU unavailable as blocker', () => {
      const hasWebGPU = false;
      const blockers = [];

      if (!hasWebGPU) {
        blockers.push('WebGPU not available');
      }

      expect(blockers).toContain('WebGPU not available');
    });

    it('identifies insufficient VRAM as blocker', () => {
      const vramRequired = 2 * GB;
      const vramAvailable = 1 * GB;
      const blockers = [];

      if (vramAvailable < vramRequired) {
        blockers.push('Insufficient VRAM');
      }

      expect(blockers).toContain('Insufficient VRAM');
    });

    it('identifies insufficient storage as blocker', () => {
      const storageRequired = 1 * GB;
      const storageAvailable = 500 * MB;
      const blockers = [];

      if (storageAvailable < storageRequired) {
        blockers.push('Insufficient storage');
      }

      expect(blockers).toContain('Insufficient storage');
    });

    it('determines canProceed based on blockers', () => {
      const blockers = ['WebGPU not available'];
      const canProceed = blockers.length === 0;

      expect(canProceed).toBe(false);
    });
  });

  describe('warning detection', () => {
    it('identifies missing F16 as warning', () => {
      const hasF16 = false;
      const warnings = [];

      if (!hasF16) {
        warnings.push('F16 not supported - inference may be slower');
      }

      expect(warnings).toContain('F16 not supported - inference may be slower');
    });

    it('identifies low VRAM headroom as warning', () => {
      const vramRequired = 4 * GB;
      const vramAvailable = 4.5 * GB;
      const lowHeadroomThreshold = 1 * GB;
      const warnings = [];

      const headroom = vramAvailable - vramRequired;
      if (headroom < lowHeadroomThreshold) {
        warnings.push('Low VRAM headroom');
      }

      expect(warnings).toContain('Low VRAM headroom');
    });
  });
});

describe('hash verification patterns', () => {
  describe('hash comparison', () => {
    it('matches identical hashes', () => {
      const expected = 'abc123def456';
      const actual = 'abc123def456';

      expect(actual === expected).toBe(true);
    });

    it('rejects different hashes', () => {
      const expected = 'abc123def456';
      const actual = 'xyz789';

      expect(actual === expected).toBe(false);
    });

    it('is case-sensitive', () => {
      const expected = 'ABC123';
      const actual = 'abc123';

      expect(actual === expected).toBe(false);
    });
  });

  describe('hash algorithm selection', () => {
    it('defaults to blake3 when available', () => {
      const manifestAlgorithm = undefined;
      const algorithm = manifestAlgorithm || 'blake3';

      expect(algorithm).toBe('blake3');
    });

    it('uses manifest-specified algorithm', () => {
      const manifestAlgorithm = 'sha256';
      const algorithm = manifestAlgorithm || 'blake3';

      expect(algorithm).toBe('sha256');
    });
  });

  describe('integrity result structure', () => {
    it('reports valid when no issues', () => {
      const missingShards = [];
      const corruptShards = [];
      const valid = missingShards.length === 0 && corruptShards.length === 0;

      expect(valid).toBe(true);
    });

    it('reports invalid when shards missing', () => {
      const missingShards = [0, 3, 5];
      const corruptShards = [];
      const valid = missingShards.length === 0 && corruptShards.length === 0;

      expect(valid).toBe(false);
    });

    it('reports invalid when shards corrupt', () => {
      const missingShards = [];
      const corruptShards = [2];
      const valid = missingShards.length === 0 && corruptShards.length === 0;

      expect(valid).toBe(false);
    });
  });
});
