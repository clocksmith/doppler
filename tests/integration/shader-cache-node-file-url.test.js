import assert from 'node:assert/strict';

const {
  clearShaderCaches,
  loadShaderSource,
} = await import('../../src/gpu/kernels/shader-cache.js');

const originalFetch = globalThis.fetch;
const originalLocation = globalThis.location;

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

  globalThis.location = { host: 'replo.id', pathname: '/run' };
  const scopedShaderCache = await import(
    `../../src/gpu/kernels/shader-cache.js?kernel-base-test=${Date.now()}`
  );
  try {
    const hostedSource = await scopedShaderCache.loadShaderSource('dequant_f16_out.wgsl');
    assert.equal(typeof hostedSource, 'string');
    assert.ok(hostedSource.length > 0);
    assert.match(hostedSource, /@compute/u);
  } finally {
    scopedShaderCache.clearShaderCaches();
  }

  console.log('shader-cache-node-file-url.test: ok');
} finally {
  clearShaderCaches();
  globalThis.fetch = originalFetch;
  if (originalLocation === undefined) {
    delete globalThis.location;
  } else {
    globalThis.location = originalLocation;
  }
}
