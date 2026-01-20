import { describe, expect, it, beforeEach } from 'vitest';

import { selectRuleValue } from '../../src/rules/rule-registry.js';
import { QK_K } from '../../src/config/schema/index.js';

describe('kernel validation', () => {
  describe('Q4K alignment constraints', () => {
    it('QK_K is 256', () => {
      expect(QK_K).toBe(256);
    });

    it('matmul selects Q4K fused regardless of K alignment when subgroups are available', () => {
      const baseContext = {
        isQ4K: true,
        hasSubgroups: true,
        fusedAllowed: true,
        useGemv: false,
        q4kVariant: 'q4_fused_batched',
        gemvVariant: null,
        matmulVariant: 'f32',
      };

      // K=1024 is aligned (1024 % 256 === 0)
      const alignedResult = selectRuleValue('kernels', 'matmul', 'matmulSelection', {
        ...baseContext,
        kAligned: true,
      });
      expect(alignedResult.useQ4KFused).toBe(true);

      // K=1152 is NOT aligned (1152 % 256 === 128) - Gemma 3 1B case
      const unalignedResult = selectRuleValue('kernels', 'matmul', 'matmulSelection', {
        ...baseContext,
        kAligned: false,
      });
      expect(unalignedResult.useQ4KFused).toBe(true);
    });

    it('fused FFN selects Q4K only when hiddenSize is aligned', () => {
      const baseContext = {
        isQ4K: true,
        fusedAllowed: true,
        batchSize: 1,
        weightDtype: 'q4k',
        useMultiOutput: false,
      };

      // hiddenSize=2048 is aligned
      const alignedResult = selectRuleValue('kernels', 'fusedFfn', 'variant', {
        ...baseContext,
        hiddenAligned: true,
      });
      expect(alignedResult).toBe('q4k');

      // hiddenSize=1152 is NOT aligned - Gemma 3 1B case
      const unalignedResult = selectRuleValue('kernels', 'fusedFfn', 'variant', {
        ...baseContext,
        hiddenAligned: false,
      });
      expect(unalignedResult).not.toBe('q4k');
      expect(unalignedResult).not.toBe('q4k_batched');
    });

    it('common model hiddenSizes alignment', () => {
      const models = [
        { name: 'Gemma 3 1B', hiddenSize: 1152, expected: false },
        { name: 'Gemma 2 2B', hiddenSize: 2304, expected: true },
        { name: 'Gemma 2 9B', hiddenSize: 3584, expected: true },
        { name: 'Llama 3 8B', hiddenSize: 4096, expected: true },
        { name: 'Qwen 3 0.6B', hiddenSize: 1024, expected: true },
        { name: 'Qwen 3 1.7B', hiddenSize: 2048, expected: true },
      ];

      for (const model of models) {
        const isAligned = model.hiddenSize % QK_K === 0;
        expect(isAligned, `${model.name} (hiddenSize=${model.hiddenSize})`).toBe(model.expected);
      }
    });
  });

  describe('matmul kernel selection rules', () => {
    it('prefers GEMV for M=1 with F16 weights and subgroups', () => {
      const result = selectRuleValue('kernels', 'matmul', 'matmulSelection', {
        isQ4K: false,
        hasSubgroups: true,
        fusedAllowed: true,
        kAligned: true,
        useGemv: true,
        q4kVariant: null,
        gemvVariant: 'gemv_subgroup_f16a',
        matmulVariant: 'f16',
      });
      expect(result.useGemv).toBe(true);
      expect(result.variant).toBe('gemv_subgroup_f16a');
    });

    it('falls back to standard matmul when no special path available', () => {
      const result = selectRuleValue('kernels', 'matmul', 'matmulSelection', {
        isQ4K: false,
        hasSubgroups: false,
        fusedAllowed: true,
        kAligned: true,
        useGemv: false,
        q4kVariant: null,
        gemvVariant: null,
        matmulVariant: 'f32',
      });
      expect(result.useQ4KFused).toBe(false);
      expect(result.useGemv).toBe(false);
    });

    it('Q4K fused requires subgroups', () => {
      const result = selectRuleValue('kernels', 'matmul', 'matmulSelection', {
        isQ4K: true,
        hasSubgroups: false,  // No subgroups
        fusedAllowed: true,
        kAligned: true,
        useGemv: false,
        q4kVariant: 'q4_fused_batched',
        gemvVariant: null,
        matmulVariant: 'f32',
      });
      expect(result.useQ4KFused).toBe(false);
    });

    it('Q4K fused respects fusedAllowed flag', () => {
      const result = selectRuleValue('kernels', 'matmul', 'matmulSelection', {
        isQ4K: true,
        hasSubgroups: true,
        fusedAllowed: false,  // Disabled by kernel path
        kAligned: true,
        useGemv: false,
        q4kVariant: 'q4_fused_batched',
        gemvVariant: null,
        matmulVariant: 'f32',
      });
      expect(result.useQ4KFused).toBe(false);
    });
  });

  describe('fused FFN kernel selection rules', () => {
    it('selects batched variant for batchSize > 1', () => {
      const result = selectRuleValue('kernels', 'fusedFfn', 'variant', {
        isQ4K: true,
        fusedAllowed: true,
        hiddenAligned: true,
        batchSize: 4,
        weightDtype: 'q4k',
        useMultiOutput: false,
      });
      expect(result).toBe('q4k_batched');
    });

    it('selects F16 variant for F16 weights', () => {
      const result = selectRuleValue('kernels', 'fusedFfn', 'variant', {
        isQ4K: false,
        fusedAllowed: true,
        hiddenAligned: true,
        batchSize: 1,
        weightDtype: 'f16',
        useMultiOutput: false,
      });
      expect(result).toBe('f16');
    });

    it('selects multi-output for small intermediate', () => {
      const result = selectRuleValue('kernels', 'fusedFfn', 'variant', {
        isQ4K: false,
        fusedAllowed: true,
        hiddenAligned: true,
        batchSize: 1,
        weightDtype: 'f32',
        useMultiOutput: true,
      });
      expect(result).toBe('multi');
    });
  });

  describe('Q4K variant selection', () => {
    it('selects multicol for M=1 with F16 activations', () => {
      const result = selectRuleValue('kernels', 'matmul', 'q4kFusedVariant', {
        useF16A: true,
        useF16Out: false,
        isM1: true,
      });
      expect(result).toBe('q4_fused_multicol_f16a');
    });

    it('selects batched for M>1 with F16 activations', () => {
      const result = selectRuleValue('kernels', 'matmul', 'q4kFusedVariant', {
        useF16A: true,
        useF16Out: false,
        isM1: false,
      });
      expect(result).toBe('q4_fused_batched_f16a');
    });

    it('selects F32 variant when no F16 requested', () => {
      const result = selectRuleValue('kernels', 'matmul', 'q4kFusedVariant', {
        useF16A: false,
        useF16Out: false,
        isM1: true,
      });
      expect(result).toBe('q4_fused_multicol');
    });
  });
});
