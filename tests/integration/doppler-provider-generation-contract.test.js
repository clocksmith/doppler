import assert from 'node:assert/strict';

import { generate } from '../../src/client/doppler-provider/generation.js';

await assert.rejects(
  async () => {
    for await (const _token of generate('hello', { stopTokens: [1, 2] })) {
      // no-op
    }
  },
  /do not support stopTokens/
);

console.log('doppler-provider-generation-contract.test: ok');
