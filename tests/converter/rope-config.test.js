import assert from 'node:assert/strict';

const { buildRoPEConfig } = await import('../../src/converter/rope-config.js');

const presetInference = {
  attention: {},
  rope: {
    ropeTheta: 1000000,
    ropeLocalTheta: 10000,
    ropeScalingType: null,
    ropeScalingFactor: 1.0,
  },
};

{
  const rope = buildRoPEConfig(presetInference, {
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
  assert.equal(rope.ropeScalingType, 'linear');
  assert.equal(rope.ropeScalingFactor, 8.0);
  assert.equal(rope.ropeLocalScalingType, null);
  assert.equal(rope.ropeLocalScalingFactor, 1.0);
}

{
  const rope = buildRoPEConfig(presetInference, {
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
  assert.equal(rope.ropeLocalScalingType, 'linear');
  assert.equal(rope.ropeLocalScalingFactor, 4.0);
}

{
  const rope = buildRoPEConfig(presetInference, {
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
  const rope = buildRoPEConfig(presetInference, {
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
  const rope = buildRoPEConfig(presetInference, {});
  assert.equal(rope.ropeTheta, 1000000);
  assert.equal(rope.ropeLocalTheta, 10000);
  assert.equal(rope.ropeScalingType, null);
  assert.equal(rope.ropeScalingFactor, 1.0);
  assert.equal(rope.ropeLocalScalingType, null);
  assert.equal(rope.ropeLocalScalingFactor, 1.0);
}

{
  assert.throws(
    () => buildRoPEConfig(presetInference, { rope_scaling: {} }),
    /missing type\/rope_type and factor/
  );
}

{
  assert.throws(
    () => buildRoPEConfig(presetInference, {
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
