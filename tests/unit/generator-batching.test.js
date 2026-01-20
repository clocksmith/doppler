import { describe, expect, it } from 'vitest';

import { resolveBatchStop, shouldUseBatchDecode } from '../../src/inference/pipeline/generator-steps.js';

describe('generator batching helpers', () => {
  it('selects batch decode only when all gates pass', () => {
    expect(shouldUseBatchDecode({
      batchSize: 4,
      useGPU: true,
      gpuSamplingAvailable: true,
      disableMultiTokenDecode: false,
      disableCommandBatching: false,
    })).toBe(true);

    expect(shouldUseBatchDecode({
      batchSize: 1,
      useGPU: true,
      gpuSamplingAvailable: true,
      disableMultiTokenDecode: false,
      disableCommandBatching: false,
    })).toBe(false);

    expect(shouldUseBatchDecode({
      batchSize: 4,
      useGPU: false,
      gpuSamplingAvailable: true,
      disableMultiTokenDecode: false,
      disableCommandBatching: false,
    })).toBe(false);
  });

  it('resolves stop flags before token scans', () => {
    const tokens = [10, 11, 12, 13];
    const stopFlags = new Uint32Array([0, 0, 1, 0]);
    const actualCount = resolveBatchStop(tokens, stopFlags, [], null);
    expect(actualCount).toBe(3);
  });

  it('resolves stop tokens when flags are empty', () => {
    const tokens = [7, 8, 9, 10];
    const actualCount = resolveBatchStop(tokens, null, [9], null);
    expect(actualCount).toBe(3);
  });

  it('prefers earliest stop token over later stop flag', () => {
    const tokens = [5, 6, 7, 8];
    const stopFlags = new Uint32Array([0, 0, 0, 1]);
    const actualCount = resolveBatchStop(tokens, stopFlags, [6], null);
    expect(actualCount).toBe(2);
  });
});
