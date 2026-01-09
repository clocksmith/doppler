/**
 * Inference Mode Tests
 *
 * Tests inference mode detection (prefill vs decode) and batching behavior.
 * No GPU required - tests orchestration logic only.
 *
 * @module tests/unit/inference-mode.test
 */

import { describe, it, expect } from 'vitest';
import {
  crossProduct,
  configName,
  INFERENCE_MODE_DIMS,
} from '../harness/test-matrix.js';

// ============================================================================
// Pure Mode Detection Logic (extracted for testability)
// ============================================================================

/**
 * Determine inference mode from token count.
 * Prefill when processing multiple tokens, decode when single.
 */
function getInferenceMode(numTokens) {
  return numTokens === 1 ? 'decode' : 'prefill';
}

/**
 * Select batch processing strategy based on config.
 */
function selectBatchStrategy(config) {
  const { batchSize, mode, hasSubgroups = true } = config;

  if (mode === 'prefill') {
    // Prefill uses tiled kernels regardless of batch
    if (batchSize >= 64) {
      return 'tiled_batched';
    }
    return 'tiled';
  }

  // Decode mode
  if (batchSize === 1) {
    return hasSubgroups ? 'gemv_subgroup' : 'gemv';
  }
  if (batchSize <= 4) {
    return 'gemv_batched';
  }
  return 'tiled_batched';
}

/**
 * Determine workgroup dispatch strategy.
 */
function getDispatchStrategy(config) {
  const { batchSize, seqLen, numHeads, intermediateSize } = config;
  const mode = getInferenceMode(seqLen);

  if (mode === 'prefill') {
    return {
      attention: { x: Math.ceil(seqLen / 32), y: numHeads, z: batchSize },
      ffn: { x: intermediateSize, y: seqLen, z: batchSize },
      matmul: { x: 'tiled', y: 'tiled', z: batchSize },
    };
  }

  // Decode: parallelize across batch and heads
  return {
    attention: { x: numHeads, y: batchSize, z: 1 },
    ffn: { x: intermediateSize, y: batchSize, z: 1 },
    matmul: { x: 'gemv', y: batchSize, z: 1 },
  };
}

/**
 * Calculate expected memory bandwidth for mode.
 */
function estimateBandwidth(config) {
  const {
    batchSize,
    seqLen,
    hiddenSize = 2048,
    intermediateSize = 8192,
    numLayers = 24,
    kvLen = 1024,
    bytesPerElement = 4,
  } = config;

  const mode = getInferenceMode(seqLen);

  // Simplified bandwidth estimate (actual is more complex)
  if (mode === 'prefill') {
    // Prefill: O(seqLen^2) attention, O(seqLen) FFN
    const attnBytes = batchSize * seqLen * seqLen * bytesPerElement * numLayers;
    const ffnBytes = batchSize * seqLen * (hiddenSize + intermediateSize) * bytesPerElement * numLayers;
    return attnBytes + ffnBytes;
  }

  // Decode: O(kvLen) attention, O(1) FFN per token
  const attnBytes = batchSize * kvLen * bytesPerElement * numLayers;
  const ffnBytes = batchSize * (hiddenSize + intermediateSize) * bytesPerElement * numLayers;
  return attnBytes + ffnBytes;
}

/**
 * Check if recorder should be used based on config.
 */
function shouldUseRecorder(config) {
  const { mode, useRecorder = true, batchSize = 1 } = config;

  // Recorder only beneficial for decode (repeated execution)
  if (mode === 'prefill') {
    return false;
  }

  // User can disable recorder
  if (!useRecorder) {
    return false;
  }

  // Recorder has overhead, only use for small batches
  return batchSize <= 8;
}

// ============================================================================
// Tests
// ============================================================================

describe('Inference Mode Detection', () => {
  it('should detect prefill for multiple tokens', () => {
    expect(getInferenceMode(64)).toBe('prefill');
    expect(getInferenceMode(256)).toBe('prefill');
    expect(getInferenceMode(2)).toBe('prefill');
  });

  it('should detect decode for single token', () => {
    expect(getInferenceMode(1)).toBe('decode');
  });

  // Parametrized tests
  describe('Mode detection matrix', () => {
    const tokenCounts = [1, 2, 8, 64, 256, 1024];

    for (const numTokens of tokenCounts) {
      it(`[numTokens=${numTokens}] => ${numTokens === 1 ? 'decode' : 'prefill'}`, () => {
        const mode = getInferenceMode(numTokens);
        expect(mode).toBe(numTokens === 1 ? 'decode' : 'prefill');
      });
    }
  });
});

describe('Batch Strategy Selection', () => {
  describe('Prefill batch strategies', () => {
    it('should use tiled for small batch prefill', () => {
      const strategy = selectBatchStrategy({ batchSize: 1, mode: 'prefill' });
      expect(strategy).toBe('tiled');
    });

    it('should use tiled_batched for large batch prefill', () => {
      const strategy = selectBatchStrategy({ batchSize: 64, mode: 'prefill' });
      expect(strategy).toBe('tiled_batched');
    });
  });

  describe('Decode batch strategies', () => {
    it('should use gemv_subgroup for single-token decode with subgroups', () => {
      const strategy = selectBatchStrategy({
        batchSize: 1,
        mode: 'decode',
        hasSubgroups: true,
      });
      expect(strategy).toBe('gemv_subgroup');
    });

    it('should use gemv for single-token decode without subgroups', () => {
      const strategy = selectBatchStrategy({
        batchSize: 1,
        mode: 'decode',
        hasSubgroups: false,
      });
      expect(strategy).toBe('gemv');
    });

    it('should use gemv_batched for small batch decode', () => {
      const strategy = selectBatchStrategy({ batchSize: 4, mode: 'decode' });
      expect(strategy).toBe('gemv_batched');
    });

    it('should use tiled_batched for large batch decode', () => {
      const strategy = selectBatchStrategy({ batchSize: 16, mode: 'decode' });
      expect(strategy).toBe('tiled_batched');
    });
  });

  // Parametrized tests
  describe('Batch strategy matrix', () => {
    const configs = crossProduct({
      batchSize: [1, 4, 8, 16, 64],
      mode: ['prefill', 'decode'],
      hasSubgroups: [true, false],
    });

    for (const config of configs) {
      it(`[${configName(config)}]`, () => {
        const strategy = selectBatchStrategy(config);
        expect(strategy).toBeDefined();
        expect(typeof strategy).toBe('string');

        // Verify constraints
        if (config.mode === 'prefill') {
          expect(strategy).toMatch(/tiled/);
        }
        if (config.mode === 'decode' && config.batchSize === 1) {
          expect(strategy).toMatch(/gemv/);
        }
      });
    }
  });
});

describe('Dispatch Strategy', () => {
  it('should parallelize across seqLen for prefill attention', () => {
    const dispatch = getDispatchStrategy({
      batchSize: 1,
      seqLen: 64,
      numHeads: 8,
      intermediateSize: 4096,
    });

    expect(dispatch.attention.x).toBe(2); // ceil(64/32)
    expect(dispatch.attention.y).toBe(8); // numHeads
    expect(dispatch.attention.z).toBe(1); // batchSize
  });

  it('should parallelize across numHeads for decode attention', () => {
    const dispatch = getDispatchStrategy({
      batchSize: 1,
      seqLen: 1,
      numHeads: 8,
      intermediateSize: 4096,
    });

    expect(dispatch.attention.x).toBe(8); // numHeads
    expect(dispatch.attention.y).toBe(1); // batchSize
    expect(dispatch.attention.z).toBe(1);
  });

  it('should use gemv strategy for decode matmul', () => {
    const dispatch = getDispatchStrategy({
      batchSize: 4,
      seqLen: 1,
      numHeads: 8,
      intermediateSize: 4096,
    });

    expect(dispatch.matmul.x).toBe('gemv');
    expect(dispatch.matmul.y).toBe(4); // batchSize
  });
});

describe('Bandwidth Estimation', () => {
  it('should estimate higher bandwidth for prefill (O(n^2) attention)', () => {
    const prefillBW = estimateBandwidth({
      batchSize: 1,
      seqLen: 256,
      kvLen: 256,
    });
    const decodeBW = estimateBandwidth({
      batchSize: 1,
      seqLen: 1,
      kvLen: 256,
    });

    // Prefill processes 256 tokens, decode processes 1
    // Prefill should have much higher bandwidth
    expect(prefillBW).toBeGreaterThan(decodeBW);
  });

  it('should scale linearly with batch size', () => {
    const bw1 = estimateBandwidth({ batchSize: 1, seqLen: 1, kvLen: 256 });
    const bw4 = estimateBandwidth({ batchSize: 4, seqLen: 1, kvLen: 256 });

    expect(bw4).toBe(bw1 * 4);
  });
});

describe('Recorder Usage', () => {
  it('should use recorder for decode with small batch', () => {
    expect(shouldUseRecorder({ mode: 'decode', batchSize: 1 })).toBe(true);
    expect(shouldUseRecorder({ mode: 'decode', batchSize: 4 })).toBe(true);
  });

  it('should not use recorder for prefill', () => {
    expect(shouldUseRecorder({ mode: 'prefill', batchSize: 1 })).toBe(false);
  });

  it('should not use recorder for large batch decode', () => {
    expect(shouldUseRecorder({ mode: 'decode', batchSize: 16 })).toBe(false);
  });

  it('should respect useRecorder flag', () => {
    expect(shouldUseRecorder({ mode: 'decode', batchSize: 1, useRecorder: false })).toBe(false);
  });

  // Parametrized tests
  describe('Recorder usage matrix', () => {
    const configs = crossProduct(INFERENCE_MODE_DIMS);

    for (const config of configs) {
      it(`[${configName(config)}]`, () => {
        const shouldRecord = shouldUseRecorder({
          mode: config.mode,
          batchSize: config.batchSize,
          useRecorder: config.useRecorder,
        });
        expect(typeof shouldRecord).toBe('boolean');

        // Verify constraints
        if (config.mode === 'prefill') {
          expect(shouldRecord).toBe(false);
        }
        if (!config.useRecorder) {
          expect(shouldRecord).toBe(false);
        }
      });
    }
  });
});

describe('Mode Transition', () => {
  /**
   * Simulate mode transition during generation.
   */
  function simulateGeneration(config) {
    const { promptLength, maxTokens } = config;
    const modes = [];

    // Prefill phase (process prompt)
    modes.push({
      step: 0,
      mode: 'prefill',
      numTokens: promptLength,
    });

    // Decode phase (generate tokens one by one)
    for (let i = 1; i <= maxTokens; i++) {
      modes.push({
        step: i,
        mode: 'decode',
        numTokens: 1,
      });
    }

    return modes;
  }

  it('should start with prefill then switch to decode', () => {
    const modes = simulateGeneration({ promptLength: 64, maxTokens: 10 });

    expect(modes[0].mode).toBe('prefill');
    expect(modes[0].numTokens).toBe(64);

    for (let i = 1; i < modes.length; i++) {
      expect(modes[i].mode).toBe('decode');
      expect(modes[i].numTokens).toBe(1);
    }
  });

  it('should have correct number of steps', () => {
    const modes = simulateGeneration({ promptLength: 32, maxTokens: 100 });

    // 1 prefill + maxTokens decode steps
    expect(modes.length).toBe(101);
  });
});

describe('Pre-built Inference Mode Configs', () => {
  const configs = crossProduct(INFERENCE_MODE_DIMS);

  for (const config of configs) {
    it(`[${configName(config)}]`, () => {
      const mode = config.mode;
      const batchSize = config.batchSize;

      // All configs should produce valid strategies
      const strategy = selectBatchStrategy({ batchSize, mode });
      expect(strategy).toBeDefined();

      // Check recorder compatibility
      const useRecorder = shouldUseRecorder({
        mode,
        batchSize,
        useRecorder: config.useRecorder,
      });
      expect(typeof useRecorder).toBe('boolean');
    });
  }
});
