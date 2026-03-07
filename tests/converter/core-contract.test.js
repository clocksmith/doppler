import assert from 'node:assert/strict';

import { transformTensorBytes } from '../../src/converter/core.js';

assert.throws(
  () => transformTensorBytes(
    {
      name: 'model.layers.0.self_attn.q_proj.weight',
      dtype: 'F16',
      shape: [1, 1],
    },
    new Uint8Array(2),
    {
      q4kLayout: 'diagonal',
    }
  ),
  /converter\.quantization\.q4kLayout must be "row" or "col"/
);

console.log('core-contract.test: ok');
