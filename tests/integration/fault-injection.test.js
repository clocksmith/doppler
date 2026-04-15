import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createFaultInjector } from '../../src/client/fault-injection.js';

describe('createFaultInjector', () => {
  it('does not inject when not enabled', () => {
    const injector = createFaultInjector({});
    assert.equal(injector.shouldInject('generate'), false);
  });

  it('does not inject when enabled is false', () => {
    const injector = createFaultInjector({
      diagnostics: { faultInjection: { enabled: false } },
    });
    assert.equal(injector.shouldInject('generate'), false);
  });

  it('injects when enabled with default probability', () => {
    const injector = createFaultInjector({
      diagnostics: { faultInjection: { enabled: true } },
    });
    // With probability 1 (default), should always inject
    assert.equal(injector.shouldInject('generate'), true);
  });

  it('respects stage filtering', () => {
    const injector = createFaultInjector({
      diagnostics: { faultInjection: { enabled: true, stage: 'prefill' } },
    });
    assert.equal(injector.shouldInject('prefill'), true);
    assert.equal(injector.shouldInject('generate'), false);
  });

  it('wildcard stage matches any stage', () => {
    const injector = createFaultInjector({
      diagnostics: { faultInjection: { enabled: true, stage: '*' } },
    });
    assert.equal(injector.shouldInject('generate'), true);
    assert.equal(injector.shouldInject('prefill'), true);
    assert.equal(injector.shouldInject('decode'), true);
  });

  it('creates error with __dopplerFaultInjected flag', () => {
    const injector = createFaultInjector({
      diagnostics: { faultInjection: { enabled: true, failureCode: 'DOPPLER_GPU_TIMEOUT' } },
    });
    const error = injector.createInjectedError();
    assert.ok(error instanceof Error);
    assert.equal(error.__dopplerFaultInjected, true);
    assert.equal(error.code, 'DOPPLER_GPU_TIMEOUT');
    assert.ok(error.message.includes('DOPPLER_GPU_TIMEOUT'));
  });

  it('uses default failureCode when not specified', () => {
    const injector = createFaultInjector({
      diagnostics: { faultInjection: { enabled: true } },
    });
    const error = injector.createInjectedError();
    assert.equal(error.code, 'DOPPLER_GPU_OOM');
  });

  it('respects probability of 0', () => {
    const injector = createFaultInjector({
      diagnostics: { faultInjection: { enabled: true, probability: 0 } },
    });
    // probability 0 should never inject
    let injected = false;
    for (let i = 0; i < 100; i++) {
      if (injector.shouldInject('generate')) {
        injected = true;
        break;
      }
    }
    assert.equal(injected, false);
  });

  it('clamps probability between 0 and 1', () => {
    const injector = createFaultInjector({
      diagnostics: { faultInjection: { enabled: true, probability: 5 } },
    });
    // Should clamp to 1, always inject
    assert.equal(injector.shouldInject('generate'), true);
  });
});
