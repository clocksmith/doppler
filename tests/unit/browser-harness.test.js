import { describe, it, expect, afterEach } from 'vitest';
import {
  loadRuntimeConfigFromUrl,
  applyRuntimePreset,
  runBrowserSuite,
} from '../../src/inference/browser-harness.js';
import { getRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
  resetRuntimeConfig();
});

describe('browser-harness runtime config loading', () => {
  it('loads runtime config from url with runtime wrapper', async () => {
    global.fetch = async () => ({
      ok: true,
      json: async () => ({
        runtime: {
          inference: {
            batching: { maxTokens: 256 },
          },
          shared: { tooling: { intent: 'verify' } },
        },
      }),
    });

    const { runtime } = await loadRuntimeConfigFromUrl('https://example.com/runtime.json');
    expect(runtime.inference.batching.maxTokens).toBe(256);
  });

  it('applies runtime preset via fetch', async () => {
    global.fetch = async () => ({
      ok: true,
      json: async () => ({
        runtime: {
          inference: {
            batching: { maxTokens: 128 },
          },
          shared: { tooling: { intent: 'verify' } },
        },
      }),
    });

    await applyRuntimePreset('custom', { baseUrl: 'https://example.com/presets' });
    expect(getRuntimeConfig().inference.batching.maxTokens).toBe(128);
  });

  it('rejects inference suite without model url or id', async () => {
    await expect(runBrowserSuite({ suite: 'inference' })).rejects.toThrow('modelUrl is required');
  });
});
