import { describe, expect, it } from 'vitest';

import { selectRuleValue } from '../../src/gpu/kernels/rule-registry.js';

describe('rule registry', () => {
  it('selects simple suffix rules', () => {
    const suffix = selectRuleValue('sample', 'suffix', { useF16: true });
    expect(suffix).toBe('_f16');
  });

  it('selects softmax topk variants', () => {
    const variant = selectRuleValue('softmax', 'topkVariant', {
      inputDtype: 'f16',
      weightsDtype: 'f16',
    });
    expect(variant).toBe('fused_f16_w16');
  });

  it('selects matmul q4k fused variants', () => {
    const variant = selectRuleValue('matmul', 'q4kFusedVariant', {
      useF16A: true,
      useF16Out: false,
      isM1: true,
    });
    expect(variant).toBe('q4_fused_multicol_f16a');
  });

  it('resolves templated attention variants', () => {
    const variant = selectRuleValue('attention', 'variant', {
      tier: 'tiled_large',
      useF16KV: false,
      canUseChunked: false,
      canUseDecodeSubgroup: false,
      base: 'prefill',
      suffix: '_f16',
      chunkedVariant: 'decode_chunked_f16',
    });
    expect(variant).toBe('prefill_f16');
  });
});
