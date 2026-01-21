import { describe, it, expect, afterEach, vi } from 'vitest';
import { createRemoteTensorSource } from '../../src/browser/tensor-source-download.js';

vi.mock('../../src/storage/backends/idb-store.js', async () => {
  const { createMemoryStore } = await import('../../src/storage/backends/memory-store.js');
  return {
    createIdbStore: () => createMemoryStore({ maxBytes: 64 * 1024 * 1024 }),
  };
});

vi.mock('../../src/storage/quota.js', () => ({
  isOPFSAvailable: () => false,
  isIndexedDBAvailable: () => true,
}));

function makeHeaders(entries) {
  return {
    get: (key) => entries[key.toLowerCase()] ?? null,
  };
}

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
  vi.clearAllMocks();
});

describe('tensor-source-download', () => {
  it('falls back to download when range is unsupported', async () => {
    const data = new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2]);

    global.fetch = vi.fn(async (_url, options) => {
      if (options?.method === 'HEAD') {
        return {
          ok: true,
          status: 200,
          headers: makeHeaders({
            'accept-ranges': 'none',
            'content-length': String(data.length),
            'content-encoding': '',
          }),
        };
      }
      return {
        ok: true,
        status: 200,
        headers: makeHeaders({
          'content-length': String(data.length),
        }),
        arrayBuffer: async () => data.buffer,
      };
    });

    const result = await createRemoteTensorSource('https://example.com/model.bin');
    expect(result.supportsRange).toBe(false);
    expect(result.size).toBe(data.length);

    const buffer = await result.source.readRange(2, 3);
    expect(Array.from(new Uint8Array(buffer))).toEqual([7, 6, 5]);
  });

  it('throws when download fallback is disabled', async () => {
    const data = new Uint8Array([1, 2, 3, 4]);

    global.fetch = vi.fn(async (_url, options) => {
      if (options?.method === 'HEAD') {
        return {
          ok: true,
          status: 200,
          headers: makeHeaders({
            'accept-ranges': 'none',
            'content-length': String(data.length),
            'content-encoding': '',
          }),
        };
      }
      return {
        ok: true,
        status: 200,
        headers: makeHeaders({
          'content-length': String(data.length),
        }),
        arrayBuffer: async () => data.buffer,
      };
    });

    await expect(
      createRemoteTensorSource('https://example.com/model.bin', { allowDownloadFallback: false })
    ).rejects.toThrow('HTTP range requests not supported');
  });

  it('enforces max download size when fallback is used', async () => {
    const data = new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2]);

    global.fetch = vi.fn(async (_url, options) => {
      if (options?.method === 'HEAD') {
        return {
          ok: true,
          status: 200,
          headers: makeHeaders({
            'accept-ranges': 'none',
            'content-length': String(data.length),
            'content-encoding': '',
          }),
        };
      }
      return {
        ok: true,
        status: 200,
        headers: makeHeaders({
          'content-length': String(data.length),
        }),
        arrayBuffer: async () => data.buffer,
      };
    });

    await expect(
      createRemoteTensorSource('https://example.com/model.bin', { maxDownloadBytes: 4 })
    ).rejects.toThrow('Download exceeds limit');
  });
});
