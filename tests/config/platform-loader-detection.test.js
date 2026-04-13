import assert from 'node:assert/strict';

const {
  clearPlatformCache,
  detectPlatform,
} = await import('../../src/config/platforms/loader.js');

async function assertPlatformMatch(adapterInfo, expectedId, message) {
  const matched = await detectPlatform(adapterInfo);
  assert.equal(matched?.id, expectedId, message);
}

try {
  await assertPlatformMatch({
    vendor: 'amd',
    architecture: 'rdna-3',
    device: 'radeon-8060s-graphics-radv-strix-halo-',
    description: 'radv: Mesa 26.0.3-1ubuntu1',
  },
    'amd-rdna3',
    'AMD RDNA3 adapters with hyphenated architecture labels must resolve to amd-rdna3.'
  );

  await assertPlatformMatch({
    vendor: 'apple',
    architecture: 'metal-3',
    device: 'apple-m3',
    description: 'Metal driver on macOS',
  },
    'apple-m3',
    'Apple M3 adapters reported as metal-3 must resolve to apple-m3.'
  );
} finally {
  clearPlatformCache();
}

console.log('platform-loader-detection.test: ok');
