import assert from 'node:assert/strict';
import { applySourceTensorRules } from '../../src/converter/source-tensor-rules.js';

const sourceTensors = [
  {
    name: 'encoder.weight',
    dtype: 'F32',
    shape: [27, 4],
    offset: 128,
    size: 432,
  },
  {
    name: 'transformer_encoder.0.ffn.w12.weight',
    dtype: 'F32',
    shape: [6, 4],
    offset: 560,
    size: 96,
  },
];

const transformed = applySourceTensorRules(sourceTensors, {
  requireAll: true,
  rules: [
    {
      kind: 'rename',
      match: '^encoder\\.weight$',
      replace: 'model.embed_tokens.weight',
      expectedMatches: 1,
    },
    {
      kind: 'split',
      match: '^transformer_encoder\\.(\\d+)\\.ffn\\.w12\\.weight$',
      axis: 0,
      expectedMatches: 1,
      parts: [
        { replace: 'model.layers.$1.mlp.gate_proj.weight', size: 3 },
        { replace: 'model.layers.$1.mlp.up_proj.weight', size: 3 },
      ],
    },
  ],
});

assert.deepEqual(
  transformed.map(({ name, shape, offset, size }) => ({ name, shape, offset, size })),
  [
    {
      name: 'model.embed_tokens.weight',
      shape: [27, 4],
      offset: 128,
      size: 432,
    },
    {
      name: 'model.layers.0.mlp.gate_proj.weight',
      shape: [3, 4],
      offset: 560,
      size: 48,
    },
    {
      name: 'model.layers.0.mlp.up_proj.weight',
      shape: [3, 4],
      offset: 608,
      size: 48,
    },
  ]
);

assert.throws(
  () => applySourceTensorRules(sourceTensors, {
    requireAll: true,
    rules: [{
      kind: 'rename',
      match: '^encoder\\.weight$',
      replace: 'model.embed_tokens.weight',
      expectedMatches: 1,
    }],
  }),
  /not covered/
);

assert.throws(
  () => applySourceTensorRules(sourceTensors, {
    requireAll: false,
    rules: [{
      kind: 'rename',
      match: '^missing$',
      replace: 'unused',
      expectedMatches: 1,
    }],
  }),
  /matched 0 tensors, expected 1/
);

assert.throws(
  () => applySourceTensorRules(sourceTensors, {
    requireAll: true,
    rules: [
      {
        kind: 'rename',
        match: '^encoder\\.weight$',
        replace: 'same.weight',
        expectedMatches: 1,
      },
      {
        kind: 'rename',
        match: '^transformer_encoder.*$',
        replace: 'same.weight',
        expectedMatches: 1,
      },
    ],
  }),
  /duplicate target tensor/
);

console.log('source-tensor-rules.test: ok');
