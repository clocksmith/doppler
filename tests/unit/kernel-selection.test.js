/**
 * Kernel Selection Logic Tests
 *
 * Tests kernel variant selection across the full configuration space.
 * No GPU required - tests selection logic only.
 *
 * @module tests/unit/kernel-selection.test
 */

import { describe, it, expect } from 'vitest';
import {
  crossProduct,
  filterConfigs,
  configName,
  MATMUL_KERNEL_CONFIGS,
  isValidMatmulConfig,
  isValidAttentionConfig,
} from '../harness/test-matrix.js';

// ============================================================================
// Pure Selection Logic (extracted from kernel modules for testability)
// ============================================================================

/**
 * Select matmul kernel variant.
 * Extracted from src/gpu/kernels/matmul.js
 */
function selectMatmulKernel(options, caps = { hasF16: true, hasSubgroups: true }) {
  const {
    aDtype = 'f32',
    bDtype = 'f32',
    outputDtype = 'f32',
    preferF16 = false,
    M = 1,
    N = 1024,
  } = options;

  const isDecode = M === 1;

  // Q4K fused path
  if (bDtype === 'q4k' && caps.hasSubgroups) {
    if (N > 8192 && isDecode) {
      return 'q4_fused_multicol';
    }
    return isDecode ? 'q4_fused' : 'q4_fused_batched';
  }

  // F16 path
  if (outputDtype === 'f16' && aDtype === 'f16' && bDtype === 'f16' && preferF16 && caps.hasF16) {
    return 'f16';
  }

  // Mixed precision: F16 weights, F32 activations
  if (bDtype === 'f16' && aDtype === 'f32' && caps.hasF16) {
    return 'f16w_f32a';
  }

  // Fallback to F32
  return 'f32';
}

/**
 * Select RMSNorm kernel variant.
 * Extracted from src/gpu/kernels/rmsnorm.js
 */
function selectRMSNormKernel(options, isF16 = false, caps = { hasSubgroups: true }) {
  const { hiddenSize = 4096, residual = null } = options;
  const smallThreshold = 512;

  if (residual) {
    return 'residual';
  }

  if (isF16) {
    if (hiddenSize <= smallThreshold) {
      return 'small_f16';
    }
    return 'f16';
  }

  if (hiddenSize <= smallThreshold) {
    return 'small';
  }

  if (caps.hasSubgroups) {
    return 'subgroup';
  }

  return 'default';
}

/**
 * Select dequant kernel variant.
 * Extracted from src/gpu/kernels/dequant.js
 */
function selectDequantKernel(options, caps = { hasF16: true, hasSubgroups: true }) {
  const { useVec4 = true, outputDtype = 'f32' } = options;

  const wantsF16Out = outputDtype === 'f16' && caps.hasF16;

  if (caps.hasSubgroups) {
    if (wantsF16Out) {
      return useVec4 ? 'subgroup_vec4_f16out' : 'subgroup_f16out';
    }
    return useVec4 ? 'subgroup_vec4' : 'subgroup';
  }

  if (wantsF16Out) {
    return useVec4 ? 'shared_vec4_f16out' : 'shared_f16out';
  }

  return useVec4 ? 'shared_vec4' : 'shared';
}

// ============================================================================
// Tests
// ============================================================================

describe('Kernel Selection', () => {
  describe('Matmul Kernel Selection', () => {
    it('should select f32 kernel for f32 inputs', () => {
      const kernel = selectMatmulKernel({ aDtype: 'f32', bDtype: 'f32', outputDtype: 'f32' });
      expect(kernel).toBe('f32');
    });

    it('should select f16 kernel for f16 inputs with f16 output', () => {
      const kernel = selectMatmulKernel({
        aDtype: 'f16',
        bDtype: 'f16',
        outputDtype: 'f16',
        preferF16: true,
      });
      expect(kernel).toBe('f16');
    });

    it('should select f16w_f32a for mixed precision (f16 weights, f32 activations)', () => {
      const kernel = selectMatmulKernel({
        aDtype: 'f32',
        bDtype: 'f16',
        outputDtype: 'f32',
        preferF16: true,
      });
      expect(kernel).toBe('f16w_f32a');
    });

    it('should fall back to f32 when F16 not available', () => {
      const kernel = selectMatmulKernel(
        { aDtype: 'f16', bDtype: 'f16', outputDtype: 'f16', preferF16: true },
        { hasF16: false, hasSubgroups: true }
      );
      expect(kernel).toBe('f32');
    });

    it('should select q4_fused for Q4K with subgroups', () => {
      const kernel = selectMatmulKernel({
        aDtype: 'f32',
        bDtype: 'q4k',
        outputDtype: 'f32',
        M: 1,
      });
      expect(kernel).toBe('q4_fused');
    });

    it('should select q4_fused_batched for batched Q4K', () => {
      const kernel = selectMatmulKernel({
        aDtype: 'f32',
        bDtype: 'q4k',
        outputDtype: 'f32',
        M: 64,
      });
      expect(kernel).toBe('q4_fused_batched');
    });

    it('should select q4_fused_multicol for large vocab', () => {
      const kernel = selectMatmulKernel({
        aDtype: 'f32',
        bDtype: 'q4k',
        outputDtype: 'f32',
        M: 1,
        N: 32000, // Large vocab
      });
      expect(kernel).toBe('q4_fused_multicol');
    });

    // Parametrized tests for matmul configs
    describe('Matmul kernel matrix', () => {
      const validConfigs = MATMUL_KERNEL_CONFIGS.filter(isValidMatmulConfig);
      for (const config of validConfigs) {
        it(`[${configName(config)}]`, () => {
          const kernel = selectMatmulKernel({
            aDtype: config.aDtype,
            bDtype: config.bDtype,
            outputDtype: config.outputDtype,
            preferF16: true,
            M: config.M,
            N: config.N,
          });
          expect(kernel).toBeDefined();
          expect(typeof kernel).toBe('string');
        });
      }
    });
  });

  describe('RMSNorm Kernel Selection', () => {
    it('should select subgroup for large hidden size with subgroups', () => {
      const kernel = selectRMSNormKernel({ hiddenSize: 4096 }, false);
      expect(kernel).toBe('subgroup');
    });

    it('should select default when no subgroups', () => {
      const kernel = selectRMSNormKernel({ hiddenSize: 4096 }, false, { hasSubgroups: false });
      expect(kernel).toBe('default');
    });

    it('should select small variant for small hidden size', () => {
      const kernel = selectRMSNormKernel({ hiddenSize: 256 }, false);
      expect(kernel).toBe('small');
    });

    it('should select residual variant when residual provided', () => {
      const kernel = selectRMSNormKernel({ hiddenSize: 4096, residual: {} }, false);
      expect(kernel).toBe('residual');
    });

    it('should select f16 variant for f16 inputs', () => {
      const kernel = selectRMSNormKernel({ hiddenSize: 4096 }, true);
      expect(kernel).toBe('f16');
    });

    it('should select small_f16 for small hidden size with f16', () => {
      const kernel = selectRMSNormKernel({ hiddenSize: 256 }, true);
      expect(kernel).toBe('small_f16');
    });

    // Parametrized tests
    describe('RMSNorm kernel matrix', () => {
      const configs = crossProduct({
        hiddenSize: [256, 1024, 4096],
        isF16: [true, false],
        hasResidual: [true, false],
      });

      for (const config of configs) {
        // Skip invalid: residual with f16 (not supported)
        if (config.hasResidual && config.isF16) continue;

        it(`[${configName(config)}]`, () => {
          const options = {
            hiddenSize: config.hiddenSize,
            residual: config.hasResidual ? {} : null,
          };
          const kernel = selectRMSNormKernel(options, config.isF16);
          expect(kernel).toBeDefined();
          expect(typeof kernel).toBe('string');

          // Verify expected patterns
          if (config.isF16) expect(kernel).toMatch(/f16/);
          if (config.hasResidual) expect(kernel).toMatch(/residual/);
          if (config.hiddenSize <= 512 && !config.hasResidual) expect(kernel).toMatch(/small/);
        });
      }
    });
  });

  describe('Dequant Kernel Selection', () => {
    it('should select subgroup_vec4 with subgroups', () => {
      const kernel = selectDequantKernel({ useVec4: true, outputDtype: 'f32' });
      expect(kernel).toBe('subgroup_vec4');
    });

    it('should select shared_vec4 without subgroups', () => {
      const kernel = selectDequantKernel(
        { useVec4: true, outputDtype: 'f32' },
        { hasF16: true, hasSubgroups: false }
      );
      expect(kernel).toBe('shared_vec4');
    });

    it('should select f16out variant for f16 output', () => {
      const kernel = selectDequantKernel({ useVec4: true, outputDtype: 'f16' });
      expect(kernel).toBe('subgroup_vec4_f16out');
    });

    it('should select subgroup without vec4', () => {
      const kernel = selectDequantKernel({ useVec4: false, outputDtype: 'f32' });
      expect(kernel).toBe('subgroup');
    });

    // Parametrized tests
    describe('Dequant kernel matrix', () => {
      const configs = crossProduct({
        useVec4: [true, false],
        outputDtype: ['f32', 'f16'],
        hasSubgroups: [true, false],
      });

      for (const config of configs) {
        it(`[${configName(config)}]`, () => {
          const kernel = selectDequantKernel(
            { useVec4: config.useVec4, outputDtype: config.outputDtype },
            { hasF16: true, hasSubgroups: config.hasSubgroups }
          );
          expect(kernel).toBeDefined();

          // Verify expected patterns
          if (config.hasSubgroups) expect(kernel).toMatch(/subgroup/);
          else expect(kernel).toMatch(/shared/);
          if (config.useVec4) expect(kernel).toMatch(/vec4/);
          if (config.outputDtype === 'f16') expect(kernel).toMatch(/f16out/);
        });
      }
    });
  });
});

describe('Test Matrix Utilities', () => {
  describe('crossProduct', () => {
    it('should generate correct cross-product', () => {
      const result = crossProduct({ a: [1, 2], b: ['x', 'y'] });
      expect(result).toHaveLength(4);
      expect(result).toContainEqual({ a: 1, b: 'x' });
      expect(result).toContainEqual({ a: 1, b: 'y' });
      expect(result).toContainEqual({ a: 2, b: 'x' });
      expect(result).toContainEqual({ a: 2, b: 'y' });
    });

    it('should handle single dimension', () => {
      const result = crossProduct({ a: [1, 2, 3] });
      expect(result).toEqual([{ a: 1 }, { a: 2 }, { a: 3 }]);
    });

    it('should handle empty dimensions', () => {
      const result = crossProduct({});
      expect(result).toEqual([{}]);
    });
  });

  describe('filterConfigs', () => {
    it('should filter configs by predicate', () => {
      const configs = [{ a: 1 }, { a: 2 }, { a: 3 }];
      const result = filterConfigs(configs, (c) => c.a > 1);
      expect(result).toEqual([{ a: 2 }, { a: 3 }]);
    });
  });

  describe('isValidMatmulConfig', () => {
    it('should reject q4k with f16 activations', () => {
      expect(isValidMatmulConfig({ aDtype: 'f16', bDtype: 'q4k', outputDtype: 'f32' })).toBe(false);
    });

    it('should accept q4k with f32 activations', () => {
      expect(isValidMatmulConfig({ aDtype: 'f32', bDtype: 'q4k', outputDtype: 'f32' })).toBe(true);
    });

    it('should reject f16 output with f32 input', () => {
      expect(isValidMatmulConfig({ aDtype: 'f32', bDtype: 'f32', outputDtype: 'f16' })).toBe(false);
    });
  });

  describe('isValidAttentionConfig', () => {
    it('should reject when numHeads not divisible by numKVHeads', () => {
      expect(isValidAttentionConfig({ numHeads: 8, numKVHeads: 3 })).toBe(false);
    });

    it('should accept valid GQA config', () => {
      expect(isValidAttentionConfig({ numHeads: 8, numKVHeads: 2, seqLen: 1, kvLen: 64 })).toBe(true);
    });
  });
});
