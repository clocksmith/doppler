import { describe, it, expect, afterEach, vi } from 'vitest';

let runtimeConfig = null;

vi.mock('../../src/config/runtime.js', () => ({
  getRuntimeConfig: () => runtimeConfig,
  setRuntimeConfig: (next) => {
    runtimeConfig = { ...next };
    return runtimeConfig;
  },
}));

vi.mock('../../src/inference/browser-harness.js', () => ({
  applyRuntimePreset: vi.fn(async () => runtimeConfig),
  runBrowserSuite: vi.fn(async () => ({
    suite: 'inference',
    passed: 1,
    failed: 0,
    results: [],
    duration: 1,
    modelId: 'test-model',
    report: { modelId: 'test-model', timestamp: '2020-01-01T00:00:00.000Z' },
    reportInfo: { path: 'reports/test-model/2020-01-01T00-00-00.000Z.json' },
  })),
}));

import { DiagnosticsController } from '../../demo/diagnostics-controller.js';
import { runBrowserSuite } from '../../src/inference/browser-harness.js';

const fakeModel = {
  key: 'test-model',
  sources: {
    browser: { id: 'test-model' },
  },
};

describe('DiagnosticsController', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('requires tooling intent for diagnostics suites', async () => {
    runtimeConfig = { shared: { tooling: { intent: null } } };
    const controller = new DiagnosticsController();

    await expect(controller.runSuite(fakeModel, {
      suite: 'inference',
      runtimeConfig,
    })).rejects.toThrow('runtime.shared.tooling.intent');
    expect(runBrowserSuite).not.toHaveBeenCalled();
  });

  it('blocks bench runs without calibrate or investigate intent', async () => {
    runtimeConfig = { shared: { tooling: { intent: 'verify' } } };
    const controller = new DiagnosticsController();

    await expect(controller.runSuite(fakeModel, {
      suite: 'bench',
      runtimeConfig,
    })).rejects.toThrow('runtime.shared.tooling.intent');
    expect(runBrowserSuite).not.toHaveBeenCalled();
  });

  it('allows inference suite when intent is verify', async () => {
    runtimeConfig = { shared: { tooling: { intent: 'verify' } } };
    const controller = new DiagnosticsController();

    const result = await controller.runSuite(fakeModel, {
      suite: 'inference',
      runtimeConfig,
    });
    expect(result?.suite).toBe('inference');
    expect(runBrowserSuite).toHaveBeenCalledTimes(1);
  });
});
