import assert from 'node:assert/strict';

import { createRemoteTensorSource } from '../../src/experimental/browser/tensor-source-download.js';

const originalFetch = globalThis.fetch;

try {
  globalThis.fetch = async (_url, options = {}) => {
    assert.equal(options.method, 'HEAD');
    return new Response('', {
      status: 200,
      headers: {
        'accept-ranges': 'none',
        'content-length': '16',
      },
    });
  };

  await assert.rejects(
    () => createRemoteTensorSource('https://example.test/model.safetensors'),
    /download fallback is not explicitly enabled/
  );

  await assert.rejects(
    () => createRemoteTensorSource('https://example.test/model.safetensors', {
      allowDownloadFallback: false,
    }),
    /HTTP range requests not supported for tensor source/
  );
} finally {
  globalThis.fetch = originalFetch;
}

console.log('browser-remote-tensor-source-contract.test: ok');
