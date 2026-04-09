import assert from 'node:assert/strict';

const {
  clearPlatformCache,
  detectPlatform,
} = await import('../../src/config/platforms/loader.js');

try {
  const matched = await detectPlatform({
    vendor: 'amd',
    architecture: 'rdna-3',
    device: 'radeon-8060s-graphics-radv-strix-halo-',
    description: 'radv: Mesa 26.0.3-1ubuntu1',
  });

  assert.equal(
    matched?.id,
    'amd-rdna3',
    'AMD RDNA3 adapters with hyphenated architecture labels must resolve to amd-rdna3.'
  );
} finally {
  clearPlatformCache();
}

console.log('platform-loader-detection.test: ok');
