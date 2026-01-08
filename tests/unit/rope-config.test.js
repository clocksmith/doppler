import { describe, expect, it } from 'vitest';

import { buildRoPEConfig } from '../../src/converter/rope-config.js';

const presetInference = {
  rope: {
    ropeTheta: 10000,
    ropeScalingType: 'linear',
    ropeScalingFactor: 1.5,
  },
};

describe('buildRoPEConfig', () => {
  it('infers linear scaling when factor is present but type is missing', () => {
    const config = {
      rope_scaling: { factor: 2.0 },
    };

    const rope = buildRoPEConfig(presetInference, config);
    expect(rope.ropeScalingType).toBe('linear');
    expect(rope.ropeScalingFactor).toBe(2.0);
  });

  it('fails fast for YARN when required params are missing', () => {
    const config = {
      rope_scaling: { type: 'yarn', factor: 2.0 },
    };

    expect(() => buildRoPEConfig(presetInference, config)).toThrow(/beta_fast/i);
  });

  it('prefers HF rope_scaling and propagates YARN params', () => {
    const config = {
      rope_theta: 9999,
      rope_scaling: {
        type: 'yarn',
        factor: 2.0,
        beta_fast: 32,
        beta_slow: 1,
        original_max_position_embeddings: 4096,
      },
    };

    const rope = buildRoPEConfig(presetInference, config);
    expect(rope.ropeTheta).toBe(9999);
    expect(rope.ropeScalingType).toBe('yarn');
    expect(rope.ropeScalingFactor).toBe(2.0);
    expect(rope.yarnBetaFast).toBe(32);
    expect(rope.yarnBetaSlow).toBe(1);
    expect(rope.yarnOriginalMaxPos).toBe(4096);
  });

  it('falls back to preset when rope_scaling is absent', () => {
    const rope = buildRoPEConfig(presetInference, {});
    expect(rope.ropeScalingType).toBe('linear');
    expect(rope.ropeScalingFactor).toBe(1.5);
  });
});
