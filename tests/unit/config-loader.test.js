import { describe, expect, it } from 'vitest';

import {
  getPreset,
  listPresets,
  resolvePreset,
  detectPreset,
  resolveConfig,
} from '../../src/config/loader.js';

describe('config/loader', () => {
  describe('getPreset', () => {
    it('returns preset by ID', () => {
      const preset = getPreset('transformer');
      expect(preset).not.toBeNull();
      expect(preset?.id).toBe('transformer');
    });

    it('returns null for unknown ID', () => {
      const preset = getPreset('nonexistent-model');
      expect(preset).toBeNull();
    });

    it('returns gemma3 preset', () => {
      const preset = getPreset('gemma3');
      expect(preset).not.toBeNull();
      expect(preset?.id).toBe('gemma3');
      expect(preset?.extends).toBe('transformer');
    });
  });

  describe('listPresets', () => {
    it('returns all registered preset IDs', () => {
      const presets = listPresets();
      expect(presets).toContain('transformer');
      expect(presets).toContain('gemma2');
      expect(presets).toContain('gemma3');
      expect(presets).toContain('llama3');
      expect(presets).toContain('mixtral');
      expect(presets.length).toBeGreaterThanOrEqual(8);
    });
  });

  describe('resolvePreset', () => {
    it('returns preset without inheritance as-is', () => {
      const preset = resolvePreset('transformer');
      expect(preset.id).toBe('transformer');
      expect(preset.extends).toBeUndefined();
    });

    it('merges child over parent', () => {
      const preset = resolvePreset('gemma3');
      // Should have gemma3's own values
      expect(preset.id).toBe('gemma3');
      // Should inherit modelType from transformer
      expect(preset.modelType).toBe('transformer');
    });

    it('deep merges nested inference config', () => {
      const gemma2 = resolvePreset('gemma2');
      // Gemma2 has specific attention settings
      expect(gemma2.inference?.attention?.attnLogitSoftcapping).toBe(50.0);
      // Should also have inherited normalization from transformer
      expect(gemma2.inference?.normalization).toBeDefined();
    });

    it('child values override parent values', () => {
      const gemma3 = resolvePreset('gemma3');
      // Gemma3 overrides queryKeyNorm
      expect(gemma3.inference?.attention?.queryKeyNorm).toBe(true);
    });

    it('throws for unknown preset', () => {
      expect(() => resolvePreset('nonexistent')).toThrow('Unknown preset: nonexistent');
    });

    it('resolves multi-level inheritance', () => {
      // functiongemma extends gemma3 extends transformer
      const preset = resolvePreset('functiongemma');
      expect(preset.id).toBe('functiongemma');
      // Should have inherited from gemma3
      expect(preset.inference?.attention?.queryKeyNorm).toBe(true);
    });
  });

  describe('detectPreset', () => {
    it('detects gemma3 from architecture pattern', () => {
      const config = {};
      const presetId = detectPreset(config, 'Gemma3ForCausalLM');
      expect(presetId).toBe('gemma3');
    });

    it('detects gemma2 from architecture pattern', () => {
      const config = {};
      const presetId = detectPreset(config, 'Gemma2ForCausalLM');
      expect(presetId).toBe('gemma2');
    });

    it('detects llama3 from architecture pattern', () => {
      const config = {};
      const presetId = detectPreset(config, 'LlamaForCausalLM');
      expect(presetId).toBe('llama3');
    });

    it('detects qwen3 from architecture pattern', () => {
      const config = {};
      const presetId = detectPreset(config, 'Qwen3ForCausalLM');
      expect(presetId).toBe('qwen3');
    });

    it('detects mamba from architecture', () => {
      const config = {};
      const presetId = detectPreset(config, 'MambaForCausalLM');
      expect(presetId).toBe('mamba');
    });

    it('falls back to transformer for unknown', () => {
      const config = { model_type: 'unknown_model' };
      const presetId = detectPreset(config, 'UnknownArchitecture');
      expect(presetId).toBe('transformer');
    });

    it('detection order is deterministic - specific before generic', () => {
      // Gemma2 should be detected before generic transformer
      const config = {};
      const presetId = detectPreset(config, 'Gemma2ForCausalLM');
      expect(presetId).toBe('gemma2');
      // Not 'transformer' even though gemma2 extends transformer
    });

    it('detects deepseek from architecture', () => {
      const config = {};
      const presetId = detectPreset(config, 'DeepseekV3ForCausalLM');
      expect(presetId).toBe('deepseek');
    });
  });

  describe('resolveConfig', () => {
    it('merges manifest architecture with preset defaults', () => {
      const manifest = {
        version: 1,
        modelId: 'test-model',
        modelType: 'transformer',
        quantization: 'F16',
        shards: [],
        totalSize: 0,
        tensorsFile: 'tensors.json',
        tensorCount: 0,
        groups: {},
        architecture: {
          numLayers: 12,
          hiddenSize: 768,
          intermediateSize: 3072,
          numAttentionHeads: 12,
          vocabSize: 32000,
          maxSeqLen: 2048,
        },
      };

      const config = resolveConfig(manifest);
      expect(config.architecture.numLayers).toBe(12);
      expect(config.architecture.hiddenSize).toBe(768);
      expect(config.architecture.numKeyValueHeads).toBe(12);
      expect(config.architecture.headDim).toBe(64);
    });

    it('inherits inference config from preset', () => {
      const manifest = {
        version: 1,
        modelId: 'gemma-test',
        modelType: 'Gemma2ForCausalLM',
        quantization: 'Q4_K_M',
        shards: [],
        totalSize: 0,
        tensorsFile: 'tensors.json',
        tensorCount: 0,
        groups: {},
        architecture: {
          numLayers: 26,
          hiddenSize: 2304,
          intermediateSize: 9216,
          numAttentionHeads: 8,
          vocabSize: 256000,
          maxSeqLen: 8192,
        },
      };

      const config = resolveConfig(manifest, 'gemma2');
      expect(config.preset).toBe('gemma2');
      // Gemma2 does NOT use queryKeyNorm (that's Gemma 3)
      expect(config.inference.attention.queryKeyNorm).toBe(false);
      // Inference config structure is present
      expect(config.inference.normalization).toBeDefined();
      expect(config.inference.ffn).toBeDefined();
    });

    it('uses detected preset when not specified', () => {
      const manifest = {
        version: 1,
        modelId: 'gemma-test',
        modelType: 'Gemma2ForCausalLM',
        quantization: 'F16',
        shards: [],
        totalSize: 0,
        tensorsFile: 'tensors.json',
        tensorCount: 0,
        groups: {},
        architecture: {
          numLayers: 26,
          hiddenSize: 2304,
          intermediateSize: 9216,
          numAttentionHeads: 8,
          vocabSize: 256000,
          maxSeqLen: 8192,
        },
      };

      const config = resolveConfig(manifest);
      expect(config.preset).toBe('gemma2');
    });

    it('includes tokenizer config from preset', () => {
      const manifest = {
        version: 1,
        modelId: 'test',
        modelType: 'transformer',
        quantization: 'F16',
        shards: [],
        totalSize: 0,
        tensorsFile: 'tensors.json',
        tensorCount: 0,
        groups: {},
        architecture: {
          numLayers: 24,
          hiddenSize: 2048,
          intermediateSize: 8192,
          numAttentionHeads: 16,
          vocabSize: 32000,
          maxSeqLen: 2048,
        },
      };

      const config = resolveConfig(manifest, 'gemma3');
      expect(config.tokenizer).toBeDefined();
    });

    it('includes sampling defaults', () => {
      const manifest = {
        version: 1,
        modelId: 'test',
        modelType: 'transformer',
        quantization: 'F16',
        shards: [],
        totalSize: 0,
        tensorsFile: 'tensors.json',
        tensorCount: 0,
        groups: {},
        architecture: {
          numLayers: 24,
          hiddenSize: 2048,
          intermediateSize: 8192,
          numAttentionHeads: 16,
          vocabSize: 32000,
          maxSeqLen: 2048,
        },
      };

      const config = resolveConfig(manifest);
      expect(config.sampling).toBeDefined();
      expect(config.sampling.temperature).toBeDefined();
      expect(config.sampling.topK).toBeDefined();
    });
  });
});
