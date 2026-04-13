import assert from 'node:assert/strict';

import {
  createHttpTensorSource,
  probeHttpRange,
} from '../../src/experimental/browser/tensor-source-http.js';

const originalFetch = globalThis.fetch;

try {
  globalThis.fetch = async () => {
    throw new Error('cors blocked');
  };

  const probe = await probeHttpRange('https://example.test/model.bin');
  assert.equal(probe.ok, false);
  assert.equal(probe.status, 0);
  assert.match(String(probe.error || ''), /cors blocked/);

  await assert.rejects(
    () => createHttpTensorSource('https://example.test/model.bin'),
    /cors blocked/
  );
} finally {
  globalThis.fetch = originalFetch;
}

console.log('tensor-source-http-contract.test: ok');
