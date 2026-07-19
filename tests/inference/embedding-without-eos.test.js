import assert from 'node:assert/strict';

import { getStopTokenIds } from '../../src/inference/pipelines/text/config.js';

assert.deepEqual(
  getStopTokenIds({ modelId: 'sequence-encoder', modelType: 'embedding', eos_token_id: null }),
  []
);
assert.throws(
  () => getStopTokenIds({ modelId: 'generator', modelType: 'text', eos_token_id: null }),
  /missing eos_token_id/
);

console.log('embedding-without-eos.test: ok');
