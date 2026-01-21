import { describe, it, expect, afterEach, vi } from 'vitest';
import { runBrowserSuite } from '../../src/inference/browser-harness.js';

let savedReport = null;

vi.mock('../../src/storage/reports.js', () => ({
  saveReport: async (modelId, report) => {
    savedReport = { modelId, report };
    return { backend: 'memory', path: `reports/${modelId}/test.json` };
  },
}));

vi.mock('../../tests/kernels/browser/test-page.js', () => ({
  initGPU: async () => {},
  testHarness: {
    references: {
      matmulRef: (_A, _B, M, N) => new Float32Array(M * N),
      rmsNormRef: (input) => new Float32Array(input.length),
    },
    getGPU: async () => ({ device: {} }),
    runMatmul: async (_device, _A, _B, M, N) => new Float32Array(M * N),
    runRMSNorm: async (_device, input) => new Float32Array(input.length),
  },
}));

describe('browser harness suites', () => {
  afterEach(() => {
    savedReport = null;
    vi.clearAllMocks();
  });

  it('runs kernel suite and saves report metadata', async () => {
    const result = await runBrowserSuite({ suite: 'kernels' });

    expect(result.report).toBeTruthy();
    expect(savedReport).not.toBeNull();
    expect(savedReport.report.suite).toBe('kernels');
    expect(savedReport.report.results.length).toBeGreaterThan(0);
  });
});
