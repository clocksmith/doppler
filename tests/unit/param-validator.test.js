import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';

import { validateCallTimeOptions, validateRuntimeConfig, validateRuntimeOverrides } from '../../src/config/param-validator.js';
import { log, setLogLevel } from '../../src/debug/index.js';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';

describe('param-validator', () => {
  
  let warnSpy = null;

  beforeEach(() => {
    setLogLevel('info');
    warnSpy = vi.spyOn(log, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy?.mockRestore();
    warnSpy = null;
  });

  it('allows generation and hybrid params at call-time', () => {
    expect(() => validateCallTimeOptions({
      temperature: 0.5,
      topK: 40,
      useChatTemplate: true,
    })).not.toThrow();
  });

  it('throws for model params at call-time', () => {
    expect(() => validateCallTimeOptions({ ropeTheta: 10000 }))
      .toThrow(/ropeTheta/);
    expect(() => validateCallTimeOptions({ slidingWindow: 4096 }))
      .toThrow(/model param/);
  });

  it('throws for session params at call-time', () => {
    expect(() => validateCallTimeOptions({ batchSize: 4 }))
      .toThrow(/session param/);
    expect(() => validateCallTimeOptions({ readbackInterval: 2 }))
      .toThrow(/session param/);
    expect(() => validateCallTimeOptions({ ringTokens: 2 }))
      .toThrow(/session param/);
  });

  it('warns when runtime overrides model params', () => {
    validateRuntimeOverrides({
      inference: {
        modelOverrides: {
          rope: { ropeTheta: 10000 },
          attention: { slidingWindow: 4096 },
        },
      },
    });

    expect(warnSpy).toHaveBeenCalledWith(
      'Config',
      expect.stringContaining('Experimental')
    );
  });

  it('ignores empty model override values', () => {
    warnSpy?.mockClear();
    validateRuntimeOverrides({
      inference: {
        modelOverrides: {
          rope: { ropeTheta: null },
        },
      },
    });

    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('validates batching readback settings', () => {
    const runtime = createDopplerConfig().runtime;
    runtime.inference.batching.readbackInterval = 0;
    expect(() => validateRuntimeConfig(runtime))
      .toThrow(/readbackInterval/);

    runtime.inference.batching.readbackInterval = null;
    runtime.inference.batching.ringTokens = null;
    runtime.inference.batching.ringStop = null;
    runtime.inference.batching.ringStaging = null;
    expect(() => validateRuntimeConfig(runtime))
      .not.toThrow();
  });
});
