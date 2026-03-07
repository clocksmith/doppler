import assert from 'node:assert/strict';

import { sortTensorsByGroup } from '../../src/converter/shard-packer.js';

assert.throws(
  () => sortTensorsByGroup([
    { name: 'model.layers.0.self_attn.q_proj.weight' },
    { name: 'model.embed_tokens.weight' },
  ]),
  /sortTensorsByGroup requires an explicit modelType/
);

const sorted = sortTensorsByGroup(
  [
    { name: 'model.layers.0.self_attn.q_proj.weight' },
    { name: 'model.embed_tokens.weight' },
  ],
  'transformer'
);

assert.equal(sorted[0].name, 'model.embed_tokens.weight');
assert.equal(sorted[1].name, 'model.layers.0.self_attn.q_proj.weight');

console.log('shard-packer-contract.test: ok');
