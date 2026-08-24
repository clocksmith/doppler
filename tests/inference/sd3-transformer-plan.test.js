import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createSD3PositionPlan,
  createSD3TransformerPlan,
  resolveSD3ModulationOffsets,
  resolveSD3ModulationSegments,
} from '../../src/inference/pipelines/diffusion/sd3/plan.js';

test('SD3 transformer plan is immutable and derives geometry without GPU state', () => {
  const plan = createSD3TransformerPlan(
    {
      num_attention_heads: 4,
      attention_head_dim: 16,
      patch_size: 2,
      norm_eps: 1e-5,
      num_layers: 3,
      dual_attention_layers: [0, 2],
      attn2_layers: [1],
    },
    { backend: {} },
    [16, 12, 10]
  );

  assert.deepEqual(
    {
      hiddenSize: plan.hiddenSize,
      tokenCount: plan.tokenCount,
      gridHeight: plan.gridHeight,
      gridWidth: plan.gridWidth,
      dualAttentionLayers: plan.dualAttentionLayers,
      attn2Layers: plan.attn2Layers,
    },
    {
      hiddenSize: 64,
      tokenCount: 30,
      gridHeight: 6,
      gridWidth: 5,
      dualAttentionLayers: [0, 2],
      attn2Layers: [1],
    }
  );
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.dualAttentionLayers), true);
});

test('SD3 position and modulation plans remain JSON-safe', () => {
  const position = createSD3PositionPlan(2, 2, 16);
  assert.deepEqual(position.indices, [0, 2, 8, 10]);
  assert.equal(position.square, true);
  assert.doesNotThrow(() => JSON.stringify(position));

  assert.equal(resolveSD3ModulationSegments([576], 64, 6, 'mod'), 9);
  assert.deepEqual(resolveSD3ModulationOffsets(6, 64).ff, {
    scale: 192,
    shift: 256,
    gate: 320,
  });
  assert.throws(
    () => resolveSD3ModulationSegments(null, 64, 6, 'missing'),
    /missing shape metadata/
  );
});

