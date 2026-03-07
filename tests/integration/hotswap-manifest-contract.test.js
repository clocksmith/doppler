import assert from 'node:assert/strict';

import { verifyHotSwapManifest } from '../../src/hotswap/manifest.js';

const policy = {
  enabled: true,
  localOnly: true,
  allowUnsignedLocal: true,
  trustedSigners: [],
};

const unsignedManifest = {
  bundleId: 'bundle-1',
  version: '1.0.0',
  artifacts: [],
};

assert.deepEqual(
  await verifyHotSwapManifest(unsignedManifest, policy, {
    source: {
      kind: 'local',
      isLocal: true,
    },
  }),
  {
    ok: true,
    reason: 'Local-only unsigned manifest accepted',
  }
);

assert.deepEqual(
  await verifyHotSwapManifest(unsignedManifest, policy, {
    source: {
      kind: 'remote',
      isLocal: false,
      url: 'https://example.test/hotswap.json',
    },
  }),
  {
    ok: false,
    reason: 'Signature required',
  }
);

console.log('hotswap-manifest-contract.test: ok');
