import assert from 'node:assert/strict';

const {
  clearShaderCaches,
  loadShaderSource,
} = await import('../../src/gpu/kernels/shader-cache.js');

const originalFetch = globalThis.fetch;

try {
  clearShaderCaches();

  if (typeof originalFetch === 'function') {
    globalThis.fetch = async (input, init) => {
      const source = typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : input?.url || '';
      if (source.startsWith('file://')) {
        throw new TypeError('fetch failed');
      }
      return originalFetch(input, init);
    };
  }

  const source = await loadShaderSource('matmul_f32.wgsl');
  assert.equal(typeof source, 'string');
  assert.ok(source.length > 0);
  assert.match(source, /@compute/u);

  console.log('shader-cache-node-file-url.test: ok');
} finally {
  clearShaderCaches();
  globalThis.fetch = originalFetch;
}
