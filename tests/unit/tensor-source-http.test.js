import { describe, it, expect, afterEach, vi } from 'vitest';
import { probeHttpRange, createHttpTensorSource } from '../../src/browser/tensor-source-http.js';

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

describe('tensor-source-http', () => {
  it('detects range support via HEAD', async () => {
    global.fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      headers: makeHeaders({
        'accept-ranges': 'bytes',
        'content-length': '16',
        'content-encoding': '',
      }),
    }));

    const result = await probeHttpRange('https://example.com/model.safetensors');
    expect(result.supportsRange).toBe(true);
    expect(result.size).toBe(16);
  });

  it('marks range unsupported when content encoding is set', async () => {
    global.fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      headers: makeHeaders({
        'accept-ranges': 'bytes',
        'content-length': '16',
        'content-encoding': 'gzip',
      }),
    }));

    const result = await probeHttpRange('https://example.com/model.safetensors');
    expect(result.supportsRange).toBe(false);
  });

  it('reads ranges from a remote source', async () => {
    const data = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);

    global.fetch = vi.fn(async (_url, options) => {
      if (options?.method === 'HEAD') {
        return {
          ok: true,
          status: 200,
          headers: makeHeaders({
            'accept-ranges': 'bytes',
            'content-length': String(data.length),
            'content-encoding': '',
          }),
        };
      }
      const range = options?.headers?.Range || options?.headers?.range;
      const match = /bytes=(\\d+)-(\\d+)/.exec(range);
      const start = match ? Number.parseInt(match[1], 10) : 0;
      const end = match ? Number.parseInt(match[2], 10) : data.length - 1;
      return {
        status: 206,
        arrayBuffer: async () => data.slice(start, end + 1).buffer,
      };
    });

    const source = await createHttpTensorSource('https://example.com/model.bin');
    const buffer = await source.readRange(2, 3);
    const bytes = new Uint8Array(buffer);
    expect(Array.from(bytes)).toEqual([3, 4, 5]);
  });
});
