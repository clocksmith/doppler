import { describe, expect, it } from 'vitest';

import {
  validateTensorConfigConsistency,
  formatValidationResult,
} from '../../src/formats/rdrr/tensor-config-validator.js';

describe('validateTensorConfigConsistency', () => {
  describe('postFeedforwardNorm detection', () => {
    it('detects missing postFeedforwardNorm=true when tensor exists', () => {
      const manifest = {
        inference: {
          normalization: {
            postFeedforwardNorm: false,
          },
        },
        groups: {
          layer_0: {
            tensors: [
              'model.layers.0.post_feedforward_layernorm.weight',
            ],
          },
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(1);
      expect(result.errors[0].code).toContain('POSTFEEDFORWARDNORM');
      expect(result.errors[0].suggestion).toContain('true');
    });

    it('passes when postFeedforwardNorm=true and tensor exists', () => {
      const manifest = {
        inference: {
          normalization: {
            postFeedforwardNorm: true,
          },
        },
        groups: {
          layer_0: {
            tensors: [
              'model.layers.0.post_feedforward_layernorm.weight',
            ],
          },
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      expect(result.valid).toBe(true);
      expect(result.errors.length).toBe(0);
    });

    it('errors when postFeedforwardNorm=true but tensor missing', () => {
      const manifest = {
        inference: {
          normalization: {
            postFeedforwardNorm: true,
          },
        },
        groups: {
          layer_0: {
            tensors: [
              'model.layers.0.input_layernorm.weight',
            ],
          },
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(1);
      expect(result.errors[0].suggestion).toContain('false');
    });
  });

  describe('preFeedforwardNorm detection', () => {
    it('detects missing preFeedforwardNorm=true when tensor exists', () => {
      const manifest = {
        inference: {
          normalization: {
            preFeedforwardNorm: false,
          },
        },
        groups: {
          layer_0: {
            tensors: [
              'model.layers.0.pre_feedforward_layernorm.weight',
            ],
          },
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(1);
      expect(result.errors[0].code).toContain('PREFEEDFORWARDNORM');
    });
  });

  describe('postAttentionNorm detection', () => {
    it('detects missing postAttentionNorm=true when tensor exists', () => {
      const manifest = {
        inference: {
          normalization: {
            postAttentionNorm: false,
          },
        },
        groups: {
          layer_0: {
            tensors: [
              'model.layers.0.post_attention_layernorm.weight',
            ],
          },
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(1);
      expect(result.errors[0].code).toContain('POSTATTENTIONNORM');
    });
  });

  describe('queryKeyNorm detection', () => {
    it('detects q_norm tensor when queryKeyNorm=false', () => {
      const manifest = {
        inference: {
          attention: {
            queryKeyNorm: false,
          },
        },
        groups: {
          layer_0: {
            tensors: [
              'model.layers.0.self_attn.q_norm.weight',
              'model.layers.0.self_attn.k_norm.weight',
            ],
          },
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      // This is a warning, not an error
      expect(result.warnings.length).toBe(1);
      expect(result.warnings[0].code).toContain('QUERYKEYNORM');
    });
  });

  describe('tieWordEmbeddings detection (inverted logic)', () => {
    it('warns when tieWordEmbeddings=true but lm_head exists', () => {
      const manifest = {
        inference: {
          output: {
            tieWordEmbeddings: true,
          },
        },
        tensors: {
          'lm_head.weight': {},
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      // This is a warning (inverted logic)
      expect(result.warnings.length).toBe(1);
      expect(result.warnings[0].code).toContain('TIEWORDEMBEDDINGS');
      expect(result.warnings[0].suggestion).toContain('false');
    });

    it('passes when tieWordEmbeddings=false and lm_head exists', () => {
      const manifest = {
        inference: {
          output: {
            tieWordEmbeddings: false,
          },
        },
        tensors: {
          'lm_head.weight': {},
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      expect(result.warnings.filter(w => w.code.includes('TIEWORDEMBEDDINGS'))).toHaveLength(0);
    });
  });

  describe('multiple issues', () => {
    it('reports multiple errors/warnings', () => {
      const manifest = {
        inference: {
          normalization: {
            postFeedforwardNorm: false,
            preFeedforwardNorm: false,
          },
        },
        groups: {
          layer_0: {
            tensors: [
              'model.layers.0.post_feedforward_layernorm.weight',
              'model.layers.0.pre_feedforward_layernorm.weight',
            ],
          },
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(2);
    });
  });

  describe('empty manifest', () => {
    it('handles missing groups and tensors gracefully', () => {
      const manifest = {
        inference: {
          normalization: {},
        },
      };

      const result = validateTensorConfigConsistency(manifest);
      // Should not throw, and should be valid (no tensors = no config required)
      expect(result.valid).toBe(true);
    });
  });
});

describe('formatValidationResult', () => {
  it('formats errors with suggestions', () => {
    const result = {
      valid: false,
      errors: [{
        severity: 'error',
        code: 'TEST_ERROR',
        message: 'Test error message',
        suggestion: 'Fix it',
      }],
      warnings: [],
    };

    const output = formatValidationResult(result);
    expect(output).toContain('ERRORS:');
    expect(output).toContain('TEST_ERROR');
    expect(output).toContain('Test error message');
    expect(output).toContain('Fix it');
  });

  it('formats warnings', () => {
    const result = {
      valid: true,
      errors: [],
      warnings: [{
        severity: 'warning',
        code: 'TEST_WARNING',
        message: 'Test warning',
        suggestion: 'Consider this',
      }],
    };

    const output = formatValidationResult(result);
    expect(output).toContain('WARNINGS:');
    expect(output).toContain('TEST_WARNING');
  });

  it('shows OK when no issues', () => {
    const result = {
      valid: true,
      errors: [],
      warnings: [],
    };

    const output = formatValidationResult(result);
    expect(output).toContain('OK');
  });
});
