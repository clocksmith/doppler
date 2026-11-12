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
    architecture: 'gfx1151',
    device: 'AMD Radeon 8060S Graphics (Strix Halo)',
    description: 'RADV Vulkan',
  },
    'amd-strix-halo-gfx1151',
    'Strix Halo gfx1151 adapters must resolve to the UMA-aware profile.'
  );

  await assertPlatformMatch({
    vendor: 'amd',
    architecture: 'rdna-3',
    device: 'radeon-8060s-graphics-radv-strix-halo-',
    description: 'radv: Mesa 26.0.3-1ubuntu1',
  },
    'amd-rdna3',
    'AMD RDNA3 adapters with hyphenated architecture labels must resolve to amd-rdna3.'
  );

  // Same Strix Halo (0x1586) hardware as reported by each runtime: Dawn names
  // the family "rdna-3", Doe names the accurate "rdna-3.5". Both must route to
  // amd-rdna3 so kernel selection matches across runtimes.
  const strixHaloAdapters = {
    dawn: {
      vendor: 'amd',
      architecture: 'rdna-3',
      device: 'radeon-8060s-graphics-radv-strix-halo-',
      description: 'radv: Mesa 26.0.3-1ubuntu1',
    },
    doe: {
      vendor: 'amd',
      architecture: 'rdna-3.5',
      device: 'Radeon 8060S Graphics (RADV STRIX_HALO)',
      description: 'Radeon 8060S Graphics (RADV STRIX_HALO)',
    },
  };
  for (const [runtime, adapterInfo] of Object.entries(strixHaloAdapters)) {
    await assertPlatformMatch(
      adapterInfo,
      'amd-rdna3',
      `Strix Halo as reported by ${runtime} must route to amd-rdna3.`
    );
  }

  // Other accurate families stay out of the RDNA 3 profile.
  for (const architecture of ['rdna-4', 'rdna-2', '']) {
    await assertPlatformMatch({
      vendor: 'amd',
      architecture,
      device: 'AMD Radeon Graphics',
      description: '',
    },
      'generic',
      `AMD architecture "${architecture}" must not route to an RDNA 3 profile.`
    );
  }

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
