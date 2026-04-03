import assert from 'node:assert/strict';

const { buildRoPEConfig } = await import('../../src/converter/rope-config.js');

const converterInference = {
  attention: {},
  rope: {
    ropeTheta: 1000000,
    ropeLocalTheta: 10000,
    ropeInterleaved: false,
    mropeInterleaved: false,
    mropeSection: null,
    partialRotaryFactor: null,
    ropeScalingType: null,
    ropeScalingFactor: 1.0,
  },
};

{
  const rope = buildRoPEConfig(converterInference, {
    text_config: {
      head_dim: 256,
      global_head_dim: 512,
      rope_parameters: {
        full_attention: {
          partial_rotary_factor: 0.25,
          rope_theta: 1000000,
          rope_type: 'proportional',
        },
        sliding_attention: {
          rope_theta: 10000,
          rope_type: 'default',
        },
      },
    },
  });
  assert.equal(rope.partialRotaryFactor, 0.25);
  assert.equal(rope.ropeLocalPartialRotaryFactor, null);
  assert.equal(rope.ropeInterleaved, false);
  assert.equal(rope.ropeFrequencyBaseDim, 512);
  assert.equal(rope.ropeLocalFrequencyBaseDim, null);
}

{
  const rope = buildRoPEConfig({
    ...converterInference,
    rope: {
      ...converterInference.rope,
      ropeInterleaved: true,
    },
  }, {
    text_config: {
      head_dim: 256,
      global_head_dim: 512,
      rope_parameters: {
        full_attention: {
          partial_rotary_factor: 0.25,
          rope_theta: 1000000,
          rope_type: 'proportional',
        },
        sliding_attention: {
          rope_theta: 10000,
          rope_type: 'default',
        },
      },
    },
  });
  assert.equal(rope.ropeInterleaved, true);
}

{
  const rope = buildRoPEConfig(converterInference, {
    rope_parameters: {
      full_attention: {
        rope_type: 'linear',
        factor: 8.0,
        rope_theta: 1000000,
      },
      sliding_attention: {
        rope_type: 'default',
        rope_theta: 10000,
      },
    },
  });
  assert.equal(rope.ropeTheta, 1000000);
  assert.equal(rope.ropeLocalTheta, 10000);
  assert.equal(rope.ropeFrequencyBaseDim, null);
  assert.equal(rope.ropeLocalFrequencyBaseDim, null);
  assert.equal(rope.ropeScalingType, 'linear');
  assert.equal(rope.ropeScalingFactor, 8.0);
  assert.equal(rope.ropeLocalScalingType, null);
  assert.equal(rope.ropeLocalScalingFactor, 1.0);
}

{
  const rope = buildRoPEConfig(converterInference, {
    rope_parameters: {
      full_attention: {
        rope_type: 'linear',
        factor: 8.0,
        rope_theta: 1000000,
      },
      sliding_attention: {
        rope_type: 'linear',
        factor: 4.0,
        rope_theta: 10000,
      },
    },
  });
  assert.equal(rope.ropeScalingType, 'linear');
  assert.equal(rope.ropeScalingFactor, 8.0);
  assert.equal(rope.ropeFrequencyBaseDim, null);
  assert.equal(rope.ropeLocalFrequencyBaseDim, null);
  assert.equal(rope.ropeLocalScalingType, 'linear');
  assert.equal(rope.ropeLocalScalingFactor, 4.0);
}

{
  const rope = buildRoPEConfig(converterInference, {
    rope_parameters: {
      full_attention: {
        rope_type: 'default',
        rope_theta: 777777,
      },
      sliding_attention: {
        rope_theta: 11111,
      },
    },
  });
  assert.equal(rope.ropeTheta, 777777);
  assert.equal(rope.ropeLocalTheta, 11111);
  assert.equal(rope.ropeScalingType, null);
  assert.equal(rope.ropeScalingFactor, 1.0);
  assert.equal(rope.ropeLocalScalingType, null);
  assert.equal(rope.ropeLocalScalingFactor, 1.0);
}

{
  const rope = buildRoPEConfig(converterInference, {
    rope_scaling: {
      rope_type: 'linear',
      factor: 4,
    },
    rope_parameters: {
      full_attention: {
        rope_type: 'linear',
        factor: 8.0,
        rope_theta: 1000000,
      },
      sliding_attention: {
        rope_theta: 10000,
      },
    },
    rope_theta: 999999,
  });
  assert.equal(rope.ropeTheta, 1000000);
  assert.equal(rope.ropeLocalTheta, 10000);
  assert.equal(rope.ropeScalingType, 'linear');
  assert.equal(rope.ropeScalingFactor, 4);
  assert.equal(rope.ropeLocalScalingType, 'linear');
  assert.equal(rope.ropeLocalScalingFactor, 4);
}

{
  const rope = buildRoPEConfig(converterInference, {
    rope_parameters: {
      rope_theta: 10000000,
      mrope_interleaved: true,
      mrope_section: [11, 11, 10],
      partial_rotary_factor: 0.25,
    },
  });
  assert.equal(rope.ropeTheta, 10000000);
  assert.equal(rope.ropeLocalTheta, 10000);
  assert.equal(rope.mropeInterleaved, true);
  assert.deepEqual(rope.mropeSection, [11, 11, 10]);
  assert.equal(rope.partialRotaryFactor, 0.25);
}

{
  const rope = buildRoPEConfig(converterInference, {});
  assert.equal(rope.ropeTheta, 1000000);
  assert.equal(rope.ropeLocalTheta, 10000);
  assert.equal(rope.ropeInterleaved, false);
  assert.equal(rope.mropeInterleaved, false);
  assert.equal(rope.mropeSection, null);
  assert.equal(rope.partialRotaryFactor, null);
  assert.equal(rope.ropeScalingType, null);
  assert.equal(rope.ropeScalingFactor, 1.0);
  assert.equal(rope.ropeLocalScalingType, null);
  assert.equal(rope.ropeLocalScalingFactor, 1.0);
}

{
  assert.throws(
    () => buildRoPEConfig(converterInference, { rope_scaling: {} }),
    /missing type\/rope_type and factor/
  );
}

{
  assert.throws(
    () => buildRoPEConfig(converterInference, {
      rope_scaling: {
        rope_type: 'linear',
        factor: 4,
      },
      rope_parameters: {
        sliding_attention: {
          rope_type: 'linear',
          factor: 8,
        },
      },
    }),
    /scaling conflicts with top-level rope_scaling/
  );
}

console.log('rope-config.test: ok');
