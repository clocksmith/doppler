import { describe, expect, it } from 'vitest';

import { parseRuntimeOverridesFromURL } from '../../src/inference/test-harness.js';
import { getRuntimeConfig, resetRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';

describe('runtime config plumbing', () => {
  it('parses runtimeConfig from URL params and applies overrides', () => {
    resetRuntimeConfig();
    const params = new URLSearchParams({
      runtimeConfig: JSON.stringify({
        inference: {
          sampling: { temperature: 0.42 },
        },
      }),
    });

    const overrides = parseRuntimeOverridesFromURL(params);
    expect(overrides.runtimeConfig?.inference?.sampling?.temperature).toBe(0.42);

    setRuntimeConfig(overrides.runtimeConfig);
    const runtime = getRuntimeConfig();
    expect(runtime.inference.sampling.temperature).toBe(0.42);
  });
});
