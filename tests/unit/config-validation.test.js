import { describe, expect, it } from 'vitest';

import {
  DEFAULT_GENERATION_CONFIG,
  DEFAULT_KERNEL_TRACE_CONFIG,
  DEFAULT_CHAT_TEMPLATE_CONFIG,
  DEFAULT_TOKENIZER_DEFAULTS,
  DEFAULT_INFERENCE_DEFAULTS_CONFIG,
  DEFAULT_RUNTIME_CONFIG,
  QK_K,
  KB,
  MB,
  GB,
} from '../../src/config/schema/index.js';

import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';

describe('config schema validation', () => {
  describe('generation config completeness', () => {
    it('has all required fields', () => {
      expect(DEFAULT_GENERATION_CONFIG.useSpeculative).toBeDefined();
      expect(DEFAULT_GENERATION_CONFIG.profile).toBeDefined();
      expect(DEFAULT_GENERATION_CONFIG.benchmark).toBeDefined();
      expect(DEFAULT_GENERATION_CONFIG.disableCommandBatching).toBeDefined();
      expect(DEFAULT_GENERATION_CONFIG.disableMultiTokenDecode).toBeDefined();
    });

    it('all fields are boolean', () => {
      for (const [key, value] of Object.entries(DEFAULT_GENERATION_CONFIG)) {
        expect(typeof value, `generation.${key}`).toBe('boolean');
      }
    });

    it('defaults to non-debug mode', () => {
      expect(DEFAULT_GENERATION_CONFIG.useSpeculative).toBe(false);
      expect(DEFAULT_GENERATION_CONFIG.profile).toBe(false);
      expect(DEFAULT_GENERATION_CONFIG.benchmark).toBe(false);
    });
  });

  describe('kernel trace config completeness', () => {
    it('has all required fields', () => {
      expect(DEFAULT_KERNEL_TRACE_CONFIG.layers).toBeDefined();
      expect(DEFAULT_KERNEL_TRACE_CONFIG.breakOnAnomaly).toBeDefined();
      expect(DEFAULT_KERNEL_TRACE_CONFIG.explosionThreshold).toBeDefined();
      expect(DEFAULT_KERNEL_TRACE_CONFIG.collapseThreshold).toBeDefined();
      expect(DEFAULT_KERNEL_TRACE_CONFIG.maxSteps).toBeDefined();
    });

    it('has sensible defaults', () => {
      expect(DEFAULT_KERNEL_TRACE_CONFIG.explosionThreshold).toBe(10);
      expect(DEFAULT_KERNEL_TRACE_CONFIG.collapseThreshold).toBe(1e-6);
      expect(DEFAULT_KERNEL_TRACE_CONFIG.maxSteps).toBe(5000);
      expect(DEFAULT_KERNEL_TRACE_CONFIG.breakOnAnomaly).toBe(false);
    });
  });

  describe('chat template config completeness', () => {
    it('has enabled field', () => {
      expect(DEFAULT_CHAT_TEMPLATE_CONFIG.enabled).toBeDefined();
    });

    it('defaults to disabled', () => {
      expect(DEFAULT_CHAT_TEMPLATE_CONFIG.enabled).toBe(false);
    });
  });

  describe('tokenizer config completeness', () => {
    it('has all required fields', () => {
      expect(DEFAULT_TOKENIZER_DEFAULTS.addBosToken).toBeDefined();
      expect(DEFAULT_TOKENIZER_DEFAULTS.addEosToken).toBeDefined();
      expect(DEFAULT_TOKENIZER_DEFAULTS).toHaveProperty('addSpacePrefix');
    });

    it('addSpacePrefix defaults to null (auto-detect)', () => {
      expect(DEFAULT_TOKENIZER_DEFAULTS.addSpacePrefix).toBeNull();
    });
  });

  describe('inference defaults config structure', () => {
    it('has all required nested configs', () => {
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.batching).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.sampling).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.compute).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.tokenizer).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.largeWeights).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.kvcache).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.moe).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.speculative).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.generation).toBeDefined();
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.chatTemplate).toBeDefined();
    });

    it('generation config is wired correctly', () => {
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.generation).toBe(DEFAULT_GENERATION_CONFIG);
    });

    it('batching defaults include readback and ring settings', () => {
      const batching = DEFAULT_INFERENCE_DEFAULTS_CONFIG.batching;
      expect(batching.readbackInterval).toBeDefined();
      expect(batching.ringTokens).toBeDefined();
      expect(batching.ringStop).toBeDefined();
      expect(batching.ringStaging).toBeDefined();
      expect(typeof batching.ringTokens).toBe('number');
      expect(typeof batching.ringStop).toBe('number');
      expect(typeof batching.ringStaging).toBe('number');
    });

    it('chat template config is wired correctly', () => {
      expect(DEFAULT_INFERENCE_DEFAULTS_CONFIG.chatTemplate).toBe(DEFAULT_CHAT_TEMPLATE_CONFIG);
    });
  });

  describe('constants are single source of truth', () => {
    it('QK_K is the Q4K block size', () => {
      expect(QK_K).toBe(256);
    });

    it('byte unit constants are correct', () => {
      expect(KB).toBe(1024);
      expect(MB).toBe(1024 * 1024);
      expect(GB).toBe(1024 * 1024 * 1024);
    });
  });
});

describe('config merge validation', () => {
  describe('createDopplerConfig deep merge', () => {
    it('preserves defaults when no overrides', () => {
      const config = createDopplerConfig();
      expect(config.runtime.inference.generation.useSpeculative).toBe(false);
      expect(config.runtime.inference.generation.profile).toBe(false);
    });

    it('partial generation override preserves other fields', () => {
      const config = createDopplerConfig({
        runtime: {
          inference: {
            generation: { profile: true },
          },
        },
      });
      expect(config.runtime.inference.generation.profile).toBe(true);
      expect(config.runtime.inference.generation.benchmark).toBe(false);
      expect(config.runtime.inference.generation.useSpeculative).toBe(false);
    });

    it('partial sampling override preserves other sampling fields', () => {
      const config = createDopplerConfig({
        runtime: {
          inference: {
            sampling: { temperature: 0.5 },
          },
        },
      });
      expect(config.runtime.inference.sampling.temperature).toBe(0.5);
      expect(config.runtime.inference.sampling.topK).toBeDefined();
      expect(config.runtime.inference.sampling.topP).toBeDefined();
    });

    it('batching overrides preserve readback and ring defaults', () => {
      const config = createDopplerConfig({
        runtime: {
          inference: {
            batching: { batchSize: 4, readbackInterval: null },
          },
        },
      });
      expect(config.runtime.inference.batching.batchSize).toBe(4);
      expect(config.runtime.inference.batching.readbackInterval).toBeNull();
      expect(config.runtime.inference.batching.ringTokens).toBeDefined();
      expect(config.runtime.inference.batching.ringStop).toBeDefined();
      expect(config.runtime.inference.batching.ringStaging).toBeDefined();
    });

    it('nested MoE config merges correctly', () => {
      const config = createDopplerConfig({
        runtime: {
          inference: {
            moe: {
              routing: { maxTokensPerExpert: 16 },
            },
          },
        },
      });
      expect(config.runtime.inference.moe.routing.maxTokensPerExpert).toBe(16);
      expect(config.runtime.inference.moe.cache).toBeDefined();
    });

    it('chatTemplate override merges correctly', () => {
      const config = createDopplerConfig({
        runtime: {
          inference: {
            chatTemplate: { enabled: true },
          },
        },
      });
      expect(config.runtime.inference.chatTemplate.enabled).toBe(true);
    });
  });

  describe('runtime config structure', () => {
    it('DEFAULT_RUNTIME_CONFIG has all top-level sections', () => {
      expect(DEFAULT_RUNTIME_CONFIG.shared).toBeDefined();
      expect(DEFAULT_RUNTIME_CONFIG.loading).toBeDefined();
      expect(DEFAULT_RUNTIME_CONFIG.inference).toBeDefined();
      expect(DEFAULT_RUNTIME_CONFIG.emulation).toBeDefined();
    });

    it('shared config has debug section', () => {
      expect(DEFAULT_RUNTIME_CONFIG.shared.debug).toBeDefined();
      expect(DEFAULT_RUNTIME_CONFIG.shared.debug.trace).toBeDefined();
      expect(DEFAULT_RUNTIME_CONFIG.shared.debug.pipeline).toBeDefined();
    });
  });
});
